#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import warnings
import pickle
import copy as cp
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from plotly.subplots import make_subplots
from scipy.interpolate import (interp1d, RegularGridInterpolator, LinearNDInterpolator,
                               NearestNDInterpolator)

# import src.core.pyairy as pyairy
# from src.core.propeller import Propeller
# import src.core.matplotlib_extended as matplotlib_extended

# C++ control
# try:
#     from src.airy.build import airy_c
# except:
#     pass    


class GeneralNDInterpolator(object):
    """
    """
    def __init__(self, points, values, inner_interpolation:str):

        if not isinstance(inner_interpolation, str):
            raise ValueError("`inner_interpolation` must be a string")
        if inner_interpolation.lower() not in ["regular grid", "unstructured"]:
            raise ValueError("Value of `inner_interpolation` not recogniized")

        if inner_interpolation.lower() == "regular grid":
            self.interpolator_inside = RegularGridInterpolator(points, values, method='cubic', bounds_error=True)
        elif inner_interpolation.lower() == "unstructured":
            self.interpolator_inside = LinearNDInterpolator(points, values, rescale=True)
        self.interporalator_outside = NearestNDInterpolator(points, values, rescale=True)

    def __call__(self, *args):
        """
        """
        value = self.interpolator_inside(*args)
        mask = np.isnan(value)
        if np.sum(mask) == 0:
            return value
        else:
            outer_values = self.interporalator_outside(*args)
            results = np.zeros(value.shape)
            results[~mask] = value[~mask]
            results[mask] = outer_values[mask]
            return results


class Foilcart(object):
    """
    Foilcart class, that includes the geometric definition and the methods to operate with this
     boat
    """
    def __init__(self, tw=15, ta=25, p_atm=101325):
        """
        Instantiation method for the Foilcart class.
        @param tw: water temperature, in degree Celsius
        @param ta: air temperature, in degree Celsius
        @param p_atm: atmospheric pressure, in Pascals
        """

        ##########################################################################################
        # DEFINITION OF GEOMETRY AND OTHER REQUIRED VARIABLES ####################################
        ##########################################################################################
        # number of points used to discretize the span-wise direction
        self.n_points = {"main wing": 301, "elevator": 101, "strut": 101}

        # Calculate water properties
        density, viscosity, pv = self._ittc_water_properties(tw)
        self.properties = {"water density [kg/m3]": density,
                           "air density [kg/m3]": p_atm / (287.15 * (ta + 273.15)),
                           "atmospheric pressure [Pa]": p_atm,
                           "viscosity [Pa s]": viscosity,
                           "vapour pressure [Pa]": pv * 1e3}

        # Definition o boat properties
        self.boat = {"CoG [m]": np.array([1.10, 0, 0.40]),
                     "mass [kg]": 175 + 80,
                     "Ixx [kg m2]": 175 * 0.4 ** 2,
                     "Iyy [kg m2]": 175 * 1.2 ** 2,
                     "Izz [kg m2]": 175 * 1.2 ** 2,
                     "Jxz [kg m2]": 175 * 1.2 * 0.3 / 10}

        self.inertia_matrix = np.array([[self.boat["Ixx [kg m2]"], 0, -self.boat["Jxz [kg m2]"]],
                                        [0, self.boat["Iyy [kg m2]"], 0],
                                        [-self.boat["Jxz [kg m2]"], 0, self.boat["Izz [kg m2]"]]])

        self.ultrasound_positions = [np.array([2.814, 0.115, 0.052]),
                                     np.array([-0.036, 0.000, 0.220])]

        # Definition of hull
        self.hull = {"CD0 [-]": 0.8,
                     "waterlength [m]": 2.997,
                     "beam [m]": 0.705,
                     "draft [m]": 0.721,
                     "center of drag [m]": np.array([3/4 * 2.997, 0, 0.2])}
        self.hull["areas [m2]"] = np.array([self.hull["beam [m]"] * self.hull["draft [m]"],
                                            self.hull["waterlength [m]"] * self.hull["draft [m]"],
                                            self.hull["waterlength [m]"] * self.hull["beam [m]"]])

        # Definition of fuselage
        self.fuselage = {"CD0 [-]": 0.3,
                         "waterlength [m]": 0.259 + 0.769,
                         "radius [m]": 0.046,
                         "center of drag [m]": np.array([3/4 * 0.259 + 1/4 * (-0.769), 0, 1.690])}
        self.fuselage["areas [m2]"] = np.array([np.pi * self.fuselage["radius [m]"] ** 2,
                                                2 * np.pi * self.fuselage["radius [m]"] *
                                                self.fuselage["waterlength [m]"],
                                                2 * np.pi * self.fuselage["radius [m]"] *
                                                self.fuselage["waterlength [m]"],
                                                ])

        # Definition of propellers
        propeller_definition = {"propeller type": "Wageningen B5-75 screw series",
                                "pitch to diameter ratio": 0.85, # Updated (0.62-1.097)
                                "diameter [m]": 0.25} # Updated
        self.propellers = {"propulsive efficiency [-]": 0.7,  # Updated (0.6-0.734)
                           "rotational speed [rev/s]": 30, # Only used for constant speed or thrust
                           "longitudinal position [m]": 0.904,
                           "lateral position [m]": 0.360,
                           "depth [m]": 1.683,
                           "stbd propeller": Propeller(propeller=propeller_definition,
                                                       propeller_label="stbd propeller"),
                           "port propeller": Propeller(propeller=propeller_definition,
                                                       propeller_label="port propeller")
                           }

        # Definition of main/bow wing
        self.main_wing = {
                          'reference point [m]': np.array([0.914, 0.0, 1.654]),
                          's bounds [-]': np.array([-1.0, 1.0]),
                          'span [m]': 1.309,
                          'chord [m]': self.calc_main_wing_chord,
                          'twist [rad]': self.calc_main_wing_twist,
                          'foil': 'naca63412',
                          'flaps': {'stbd': [(0.12, 0.49), 0.453],
                                    'port': [(-0.49, -0.12), 0.453]},
                          'destroy lift': {"fuselage": (-0.08, 0.08),
                                           "stbd motor": (0.50, 0.60),
                                           "port motor": (-0.60, -0.50)}
                          }
        self.main_wing['quarter-chord [m]'] = self.main_wing_qc

        # Definition of elevator/aft wing
        self.elevator = {
                         'reference point [m]': np.array([-0.389, 0.0, 1.590]),
                         's bounds [-]': np.array([-1.0, 1.0]),
                         'span [m]': 0.698,
                         'chord [m]': self.calc_elevator_chord,
                         'twist [rad]': self.calc_elevator_twist,
                         'foil': 'naca0013',
                         'flaps': {}
                         }
        self.elevator['quarter-chord [m]'] = self.elevator_qc

        # Definition of strut
        self.strut = {
                      'reference point [m]': np.array([0.751, 0.0, 1.654]),
                      's bounds [-]': np.array([0.0, 1.0]),
                      'span [m]': 1.120,
                      'chord [m]': self.calc_strut_chord,
                      'twist [rad]': self.calc_strut_twist,
                      'foil': 'naca0013',
                      'flaps': {'lower': [(0.10, 0.32), 0.41],
                                'upper': [(0.32, 0.54), 0.41]}
                      }
        self.strut['quarter-chord [m]'] = self.strut_qc


        ##########################################################################################
        # PREPROCESS GEOMETRIES ##################################################################
        ##########################################################################################
        # Assign aliases to the various geometric designs
        self.dict_of_designs = {"main wing": self.main_wing,
                                "elevator": self.elevator,
                                "strut": self.strut}

        # Preprocess geometries
        self.wings = {}
        for label, design in self.dict_of_designs.items():
            # Use a different container to store data
            wing = {}
            # Calculate span for horizontal and vertical wings
            s0, s1 = design['s bounds [-]']
            x0 = design["quarter-chord [m]"](s0)
            x1 = design["quarter-chord [m]"](s1)
            if label != "strut":
                wing["span"] = np.abs(x1[1] - x0[1])
            else:
                wing["span"] = np.abs(x1[2] - x0[2])

            # Load the values of the lift coefficient once they have been fitted to a straight line
            storage_path = './data/' + design["foil"] + '/'
            df = pd.read_csv(storage_path + 'fitted_cl0.csv', index_col=0)
            # get list of flap deflections and hinge positions (measured from leading edge) for
            #   which the performances have been calculated
            flap_length_ratios = list(map(float, df.columns.values))
            flap_deflections = list(map(float, df.index.values))
            # Fit interpolants for the linear approximations of the lift curve. We obtain
            #   functions that provide the lift coefficient at zero AoA, the slope of the lift
            #   coefficient, and the null lift angle as a function of the the deflection of the
            #   flap and the position of its hinge
            cl0_values = df.values.T
            cl0_values[np.abs(cl0_values) < 1e-3] = 0.
            wing["cl0"] = RegularGridInterpolator((flap_length_ratios, flap_deflections),
                                                  cl0_values, method='cubic', bounds_error=True)
            df = pd.read_csv(storage_path + 'fitted_cla.csv', index_col=0)
            wing["cla"] = RegularGridInterpolator((flap_length_ratios, flap_deflections),
                                                  df.values.T, method='cubic', bounds_error=True)
            df = pd.read_csv(storage_path + 'fitted_aoanl.csv', index_col=0)
            aoa_nl_values = df.values.T
            aoa_nl_values[np.abs(aoa_nl_values) < 1e-3] = 0.
            wing["aoa nl"] = RegularGridInterpolator((flap_length_ratios, flap_deflections),
                                                     df.values.T, method='cubic',
                                                     bounds_error=True)
            # We use an arbitrary value of the position of the hinge for those sections without
            #   flap. For these section the deflection will always be zero and the position is
            #   irrelevant, but the interpolator requires to input a valid value.
            wing['reference flap length_ratio'] = float(df.columns[0])

            # Load all, raw foil data
            with open(storage_path + "all_data.pickle", "rb") as input_file:
                df_dict = pickle.load(input_file)
            # Get values of the calculated angles of attack
            all_alpha_values = set()
            all_lists_are_the_same = True
            for length, deflection in product(flap_length_ratios, flap_deflections):
                these_alpha_values = set(df_dict[(length, deflection)].alpha.values)
                all_alpha_values = all_alpha_values.union(these_alpha_values)
                all_lists_are_the_same = all_lists_are_the_same and (all_alpha_values ==
                                                                     these_alpha_values)
            all_alpha_values = sorted(all_alpha_values)
            if all_lists_are_the_same:
                # Initialize arrays (3rd order tensors) that store the values of the parasitic
                #   drag,pitch moment, hinge moment, and minimum pressure coefficients
                chosen_shape = (len(flap_length_ratios), len(flap_deflections),
                                len(all_alpha_values))
                cd0 = np.zeros(chosen_shape)
                cmy = np.zeros(chosen_shape)
                cp_min = np.zeros(chosen_shape)
                ch = np.zeros(chosen_shape)
                for pp, flap_pos_n in enumerate(flap_length_ratios):
                    for dd, flap_def_n in enumerate(flap_deflections):
                        key = (flap_pos_n, flap_def_n)

                        # If not all values of AoA are present because errors in XFoil simulation,
                        #   we interpolate the missing points
                        if len(all_alpha_values) != len(df_dict[key].alpha.values):
                            interp = interp1d(df_dict[key].alpha.values, df_dict[key].CD.values,
                                              bounds_error=False, fill_value=(df_dict[key].CD.values[0], df_dict[key].CD.values[-1]))
                            cd0_values = interp(all_alpha_values)
                            interp = interp1d(df_dict[key].alpha.values, df_dict[key].CM.values,
                                              bounds_error=False, fill_value=(df_dict[key].CM.values[0], df_dict[key].CM.values[-1]))
                            cmy_values = interp(all_alpha_values)
                            interp = interp1d(df_dict[key].alpha.values, df_dict[key].Cpmin.values,
                                              bounds_error=False, fill_value=(df_dict[key].Cpmin.values[0], df_dict[key].Cpmin.values[-1]))
                            cp_min_values = interp(all_alpha_values)
                            interp = interp1d(df_dict[key].alpha.values, df_dict[key].Chinge.values,
                                              bounds_error=False, fill_value=(df_dict[key].Chinge.values[0], df_dict[key].Chinge.values[-1]))
                            ch_values = interp(all_alpha_values)

                            alpha_set = set(all_alpha_values)
                            for value in df_dict[key].alpha.values:
                                alpha_set.discard(value)
                            print(label, key, alpha_set)
                        else:
                            cd0_values = df_dict[key].CD.values
                            cmy_values = df_dict[key].CM.values
                            cp_min_values = df_dict[key].Cpmin.values
                            ch_values = df_dict[key].Chinge.values
                        cd0[pp, dd, :] = cd0_values
                        cmy[pp, dd, :] = cmy_values
                        cp_min[pp, dd, :] = cp_min_values
                        ch[pp, dd, :] = ch_values

                # Once all data is ordered, 3d-interpolators are fitted for further use
                grid = (flap_length_ratios, flap_deflections, all_alpha_values)
                # wing["cd0"] = RegularGridInterpolator(grid, cd0, method='cubic', bounds_error=False, fill_value=None)
                # wing["cmy"] = RegularGridInterpolator(grid, cmy, method='cubic', bounds_error=False, fill_value=None)
                # wing["cp_min"] = RegularGridInterpolator(grid, cp_min, method='cubic', bounds_error=False, fill_value=None)
                # wing["ch"] = RegularGridInterpolator(grid, ch, method='cubic', bounds_error=False, fill_value=None)
                wing["cmy"] = GeneralNDInterpolator(grid, cmy, "regular grid")
                wing["cp_min"] = GeneralNDInterpolator(grid, cp_min, "regular grid")
                wing["cd0"] = GeneralNDInterpolator(grid, cd0, "regular grid")
                wing["ch"] = GeneralNDInterpolator(grid, ch, "regular grid")
            else:
                # Initialize 1d arrays of unsorted data that store the values of the parasitic
                #   drag, pitch moment, hinge moment, and minimum pressure coefficients
                points = []
                cd0 = []
                cmy = []
                cp_min = []
                ch = []
                for pp, flap_pos_n in enumerate(flap_length_ratios):
                    for dd, flap_def_n in enumerate(flap_deflections):
                        key = (flap_pos_n, flap_def_n)
                        points.extend([(flap_pos_n, flap_def_n, alpha)
                                       for alpha in df_dict[key].alpha.values])
                        cd0.extend([value for value in df_dict[key].CD.values])
                        cmy.extend([value for value in df_dict[key].CM.values])
                        cp_min.extend([value for value in df_dict[key].Cpmin.values])
                        ch.extend([value for value in df_dict[key].Chinge.values])

                # Once all data is ordered, 3d-, scattered interpolators are fitted for further use
                # wing["cd0"] = LinearNDInterpolator(points, cd0, rescale=True)
                # wing["cmy"] = LinearNDInterpolator(points, cmy, rescale=True)
                # wing["cp_min"] = LinearNDInterpolator(points, cp_min, rescale=True
                # wing["ch"] = LinearNDInterpolator(points, ch, rescale=True)
                wing["cd0"] = GeneralNDInterpolator(points, cd0, "unstructured")
                wing["cmy"] = GeneralNDInterpolator(points, cmy, "unstructured")
                wing["cp_min"] = GeneralNDInterpolator(points, cp_min, "unstructured")
                wing["ch"] = GeneralNDInterpolator(points, ch, "unstructured")

            # The inverse quarter-chord functions are initialized
            if label != "strut":
                wing["s from qc"] = self.inverse_straight_wing_qc_line
            else:
                wing["s from qc"] = self.inverse_straight_strut_qc_line
            wing["flap labels"] = [key for key in design["flaps"].keys()]

            # Definition of unit chord shape. Used only for plotting methods.
            n_foil = 101
            foil_coordinates = np.zeros((2 * n_foil, 2))
            t = 0.13
            for qq in range(n_foil):
                x = 1 - qq / (n_foil - 1)
                y = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                             + 0.2843 * x ** 3 - 0.1015 * x ** 4)
                foil_coordinates[qq, 0] = 0.25 - x
                foil_coordinates[qq, 1] = y
                x = qq / (n_foil - 1)
                y = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                             + 0.2843 * x ** 3 - 0.1015 * x ** 4)
                foil_coordinates[n_foil + qq, 0] = 0.25 - x
                foil_coordinates[n_foil + qq, 1] = -y

            if label != "strut":
                wing["unit chord foil"] = np.column_stack((foil_coordinates[:, 0],
                                                           np.zeros((2 * n_foil, )),
                                                           foil_coordinates[:, 1])).T
            else:
                wing["unit chord foil"] = np.column_stack((foil_coordinates[:, 0],
                                                           foil_coordinates[:, 1],
                                                           np.zeros((2 * n_foil,)))).T
            # Store the data for further use
            self.wings[label] = wing

        self.equation_labels = ["kinematic x", "kinematic y", "kinematic z",
                                "dot yaw", "dot pitch", "dot roll",
                                "force x", "force y", "force z",
                                "moment x", "moment y", "moment z",
                                "ground force x", "ground force y", "ground force z",
                                "ground moment x", "ground moment y", "ground moment z"
                                ]
        self.state_labels = ["x", "y", "z", "yaw", "pitch", "roll", "u", "v", "w", "p", "q", "r"]
        self.input_labels_short = {"main wing": "mw", "elevator": "elev", "rudder": "rudd", 
                                   "average rotation speed": "avg rot spd", "differential rotation speed": "diff rot spd", 
                                   "total thrust": "tot thr", "differential thrust": "diff thr"}
        self.inputs_for_this_case = None
        self.fixed_inputs = {}
    
    # Local functions/lambdas prevent pickling so they need to be here instead
    def calc_main_wing_chord(self, ss):
        return 0.200 * np.sqrt(1 - (ss / 1.) ** 2)
    def calc_elevator_chord(self, ss):
        return 0.124 * np.sqrt(1 - (ss / 1) ** 2)
    def calc_strut_chord(self, ss):
        return 0.197
    def calc_main_wing_twist(self, ss):
        return 0
    def calc_elevator_twist(self, ss):
        return 0
    def calc_strut_twist(self, ss):
        return 0
    def main_wing_qc(self, ss):
        return (self.main_wing["span [m]"] / 2) * np.array([0. * ss, 1. * ss, 0. * ss])
    def elevator_qc(self, ss):
        xx = 0.5 * (self.elevator["chord [m]"](ss) - self.elevator["chord [m]"](0))
        return (self.elevator["span [m]"] / 2) * np.array([xx, 1. * ss, 0. * ss])
    def strut_qc(self, ss):
        return self.strut["span [m]"] * np.array([0. * ss, 0. * ss, -1. * ss])
    def inverse_straight_wing_qc_line(self, yy, span):
        return 2 * yy / span
    def inverse_straight_strut_qc_line(self, zz, span):
        return -zz / span
    
    @staticmethod
    def _general_nd_interpolator(interpolator1, interpolator2, point):
        """
        """
        interpolator1(point)

    @staticmethod
    def _induced_aoa_modes(mode_no, theta):
        """Calculates the Glauert integral for a mode and a inputted point. The sum of these
        values by the coefficients of the expansion in sines of the circulation provides the
        induced angle of attack
        @param mode_no: int >=1, number of mode for which the integral is calculated
        @param theta: float in [0, pi], position where integral is calculated
        @return: float, coefficient of the "mode_no"-th mode at theta
        """
        if np.abs(theta) < 1e-10:
            g = mode_no
        elif abs(theta - np.pi) < 1e-10:
            g = mode_no * (-1) ** (mode_no + 1)
        else:
            g = np.sin(mode_no * theta) / np.sin(theta)
        return mode_no * g / 2

    def prandtl_lifting_line_preprocess(self, wing_label, span=None,
                                        n_modes=None, flap_angles=None):
        """ The preprocessing operations for the classical Prandtl lifting line method. If the
        geometry of the wing (and the deflection of the flaps and the submerged length of the
        strut are part of the geometry) is not changed, you can save time by not running it
        again.
        @param wing_label: sting, label of the wing to be calculated.
        @param span: float > 0, value of the span of the wing. It will only be inputted for
            struts, so we will assume that, if the inputted value is not None, we are dealing with
            a strut.
        @param n_modes: integer, number of sines in which the circulation will be expanded.
        @param flap_angles: dict, values of the deflections of each flap (in degrees)
        @return: dict with all the calculated data that can be useful in the future
        """

        # number of points in which the span will be discretized.
        n_points = self.n_points[wing_label]

        # Number of points must be odd
        if n_points % 2 == 0:
            raise ValueError("Number of points in which wings are discretized must be odd")

        # Assign aliases to dicts of data
        design = self.dict_of_designs[wing_label]
        wing = self.wings[wing_label]

        # We assume that we are dealing with a strut if the span is inputted
        is_strut = span is not None

        # Span is calculated
        if not is_strut:
            span = wing["span"]
        else:
            # If we are dealing with a strut we duplicate the span
            span *= 2
            if span > 2 * wing["span"]:
                print("span", span)
                print("wing span", wing["span"])
                raise ValueError("Strut is fully submerged. Hull touches the water.")
            elif span < 0:
                raise ValueError("Strut is out of the water. Some wing must be also out of it")

        # If the number of modes is not specified we assume a value
        if n_modes is None:
            n_modes = int(1 + (n_points - 1) / 5)

        # Discretize co-ordinates in the Prandtl co-ordinate system and their physical
        #   and parametric counterparts
        theta = np.linspace(0, np.pi, n_points)
        if not is_strut:
            spanwise_x = np.cos(theta) * span / 2
        else:
            spanwise_x = -np.cos(theta) * span / 2
        s = wing["s from qc"](spanwise_x, wing["span"])
        # To symmetrically extend wings a mask is created: if a value is True the value must be
        #   extended, otherwise it must be created.
        if is_strut:
            symmetry_mask = s < -1e-10
            index_of_mirror = int((n_points - 1) / 2)
            if symmetry_mask[index_of_mirror]:
                raise ValueError("Check definition of {} because it cannot be symmetrically "
                                 "extended".format(wing_label))
            if (np.any(symmetry_mask[:index_of_mirror + 1]) or
                    not np.all(symmetry_mask[index_of_mirror + 1:n_points])):
                raise ValueError("Something unexpected happened when trying to extend "
                                 "symmetrically the wing {}".format(wing_label))
        else:
            index_of_mirror = n_points - 1
            symmetry_mask = np.full((n_points,), False)

        # Get flap angles or, if not inputted, assign zero to all of them
        if flap_angles is not None:
            if not isinstance(flap_angles, dict):
                raise ValueError("`flap_angles` must be a dictionary")
            elif design["flaps"].keys() != flap_angles.keys():
                print(design["flaps"].keys(), flap_angles.keys())
                raise ValueError("The keys of `flap_angles` must be equal to the labels assigned "
                                 "to the flaps of the wing")
        else:
            flap_angles = {flap_label: 0 for flap_label in design["flaps"].keys()}

        # Calculate values of the chords, slope of lifting coefficients, and null-lift angles of
        #   attack at the nodes of the discretized span. We also calculate the positions of the
        chords = np.empty(n_points,)
        cla = np.empty(n_points, )
        aoa_nl = np.empty(n_points,)
        x = np.empty((n_points, 3))
        for pp in range(index_of_mirror+1):
            chords[pp] = design['chord [m]'](s[pp])
            cla[pp] = wing["cla"]([wing['reference flap length_ratio'], 0])
            aoa_nl[pp] = wing["aoa nl"]([wing['reference flap length_ratio'], 0])
            x[pp, :] = design["quarter-chord [m]"](s[pp])
        if is_strut:
            chords = self._extend_symmetrically(chords, "vector")
            cla = self._extend_symmetrically(cla, "vector")
            aoa_nl = self._extend_symmetrically(aoa_nl, "vector")
            x = self._extend_symmetrically(x, "x")

        # Calculate some properties at root
        y_root = np.cos(np.pi / 2) * span / 2
        s_root = wing["s from qc"](y_root, wing["span"])
        root_chord = design['chord [m]'](s_root)

        # The calculation of the slope of the lift coefficient and the null-lift angle performed
        #   above did not account for the presence of deflected flaps. Now we calculate their
        #   effect over those variables
        flap_mask = np.array([None for _ in range(n_points)])
        flap_ratio = wing['reference flap length_ratio'] * np.ones(n_points, )
        for flap_label, flap_data in design["flaps"].items():
            flap_interval = flap_data[0]
            flap_length_at_root = flap_data[1]
            for pp in range(index_of_mirror+1):
                if flap_interval[0] <= s[pp] <= flap_interval[1]:
                    flap_mask[pp] = flap_label
                    cf = flap_length_at_root * root_chord - 3 / 4 * (root_chord - chords[pp])
                    flap_ratio[pp] = 1 - cf / chords[pp]
                    delta = flap_angles[flap_label]
                    try:
                        cla[pp] = wing["cla"]([flap_ratio[pp], delta])
                        aoa_nl[pp] = wing["aoa nl"]([flap_ratio[pp], delta])
                    except Exception as e:
                        values = (flap_ratio[pp], delta, wing_label)
                        print('Flap length, {:.3f}, or flap deflection, {:.3f}, '
                                         'are out of bounds ({})'.format(*values))
                        raise e
            if is_strut:
                flap_ratio = self._extend_symmetrically(flap_ratio, "vector")
                cla = self._extend_symmetrically(cla, "vector")
                aoa_nl = self._extend_symmetrically(aoa_nl, "vector")
                flap_mask = self._extend_symmetrically(flap_mask, "flap mask")

        # Destroy lift if certain regions if required
        if "destroy lift" in design.keys():
            for region_label, region_range in design["destroy lift"].items():
                # All point inside the interval won't produce lift
                mask = np.logical_and(s >= region_range[0], s <= region_range[1])
                # Edges of the interval must be checked individually
                error = np.abs(region_range[0] - s)
                index = np.argmin(error)
                if error[index] < 1e-10:
                    mask[index] = True
                error = np.abs(region_range[1] - s)
                index = np.argmin(error)
                if error[index] < 1e-10:
                    mask[index] = True
                # Set lift for any AoA equal to zero
                cla[mask] = 0
                aoa_nl[mask] = 0

        # We calculate the values of the modes at the nodes for further reconstruction and
        #   evaluation of forces and moments
        sine_modes = np.zeros((n_points, n_modes))
        for nn in range(n_modes):
            sine_modes[:, nn] = np.sin((nn + 1) * theta)

        # We also calculate the integral weights used to numerically integrate the distributions
        #   of forces and moments
        if not is_strut:
            int_w = self._calculate_integral_weights(theta[1] - theta[0], n_points)
            wing_integral_weights = 0.5 * span * sine_modes[:, 0] * int_w
        else:
            int_w = self._calculate_integral_weights(theta[1] - theta[0], sum(~symmetry_mask))
            wing_integral_weights = np.zeros(n_points)
            chosen_modes = sine_modes[~symmetry_mask, 0]
            wing_integral_weights[:sum(~symmetry_mask)] = 0.5 * span * chosen_modes * int_w
            # raise NotImplementedError('DO IT!')

        hinge_integral_weights = {}
        for key in flap_angles:
            hinge_mask = flap_mask == key
            if np.sum(hinge_mask) != 0:
                int_w = self._calculate_integral_weights(theta[1] - theta[0], np.sum(hinge_mask))
                hinge_integral_weights[key] = 0.5 * span * sine_modes[hinge_mask, 0] * int_w
            else:
                hinge_integral_weights[key] = np.array([])

        # Calculate some geometric parameters of the wing
        area = chords.dot(wing_integral_weights)
        average_chord = area / span
        aspect_ratio = span ** 2 / area

        # Finally, we calculate the matrix of the Prandtl LL system, the vector that multiplies
        #   the effective angle of attack to provide the independent term of the system,
        #   and the matrix that provides the induced AoA if pre-multiplied by the circulation
        #   coefficients
        circulation_matrix = np.zeros((n_points, n_modes))
        aoa_i_matrix = np.zeros((n_points, n_modes))
        circulation_vector = np.zeros((n_points,))
        for pp in range(n_points):
            f = 0.5 * chords[pp] * cla[pp] / span
            for qq in range(n_modes):
                g = self._induced_aoa_modes(qq + 1, theta[pp])
                aoa_i_matrix[pp, qq] = g
                circulation_matrix[pp, qq] = np.sin((qq+1) * theta[pp]) + f * g
            circulation_vector[pp] = f

        preprocessed_prandtl = {'wing label': wing_label,
                                '#points': n_points,
                                '#modes': n_modes,
                                'theta': theta,
                                'span': span,
                                'span-wise x': spanwise_x,
                                's': s,
                                'x': x,
                                'chords': chords,
                                'flap mask': flap_mask,
                                'flap ratio': flap_ratio,
                                'flap angles': flap_angles,
                                'cla': cla,
                                'aoa nl': aoa_nl,
                                'circulation matrix': circulation_matrix,
                                'circulation vector': circulation_vector,
                                'aoa_i modes': aoa_i_matrix,
                                'is it a strut': is_strut,
                                'symmetry mask': symmetry_mask,
                                'index of mirror': index_of_mirror,
                                'sine modes': sine_modes,
                                'wing integral weights': wing_integral_weights,
                                'hinge integral weights': hinge_integral_weights,
                                'area': area,
                                'average chord': average_chord,
                                'aspect ratio': aspect_ratio}

        return preprocessed_prandtl

    def prandtl_lifting_line_solve(self, preprocessed_prandtl, speed, aoa_geo, aoa_flow,
                                   aoa_twist):
        """ Solves the classical Prandtl lifting line formulation once the geometry has already
         been preprocessed.
        @param preprocessed_prandtl: dictionary of the variables required to preprocess the
            geometry and solve an specific case.
        @param speed: scalar or NumPy array of speeds at the nodes of the geometry
        @param aoa_geo: NumPy array of the geometric angles of the wing at each section
        @param aoa_flow:
        @param aoa_twist:
        @return:
        """

        # Unpack some values and assign shorter aliases
        density = self.properties["water density [kg/m3]"]
        static_p = self.properties["atmospheric pressure [Pa]"]
        pv = self.properties["vapour pressure [Pa]"]
        viscosity = self.properties["viscosity [Pa s]"]

        dynamic_pressure = 0.5 * density * speed ** 2

        n_points = preprocessed_prandtl["#points"]

        span = preprocessed_prandtl["span"]
        area = preprocessed_prandtl["area"]
        chords = preprocessed_prandtl["chords"]
        aoa_nl = preprocessed_prandtl['aoa nl']
        wing_label = preprocessed_prandtl["wing label"]
        sine_modes = preprocessed_prandtl['sine modes']
        is_strut = preprocessed_prandtl['is it a strut']
        circulation_matrix = preprocessed_prandtl['circulation matrix']
        circulation_vector = preprocessed_prandtl['circulation vector']
        wing_integral_weights = preprocessed_prandtl["wing integral weights"]
        hinge_integral_weights = preprocessed_prandtl["hinge integral weights"]

        if is_strut:
            aoa_geo = self._extend_symmetrically(aoa_geo, "vector")
            aoa_flow = self._extend_symmetrically(aoa_flow, "vector")
            aoa_twist = self._extend_symmetrically(aoa_twist, "vector")

        # Solve linear system
        b = circulation_vector * (aoa_geo + aoa_flow + aoa_twist - aoa_nl)
        circulation_amplitudes = np.linalg.solve(circulation_matrix.T.dot(circulation_matrix),
                                                 circulation_matrix.T.dot(b))
        circulation_amplitudes *= span * speed
        circulation = sine_modes.dot(circulation_amplitudes)

        # Calculate induced angles of attack and the effective angle of attack at each section
        aoa_i = -preprocessed_prandtl["aoa_i modes"].dot(circulation_amplitudes) / (span * speed)
        aoa_eff = np.degrees(aoa_geo + aoa_flow + aoa_twist + aoa_i)

        # Get flap angles at each section
        flap_angles = np.zeros(n_points)
        for key, value in preprocessed_prandtl["flap angles"].items():
            flap_angles[preprocessed_prandtl["flap mask"] == key] = value
        flap_ratio = preprocessed_prandtl["flap ratio"]

        # Calculate values of some hydrodynamic coefficients for the solution, namely, parasitic
        #  drag (friction + pressure), pitch moment, minimum pressure, and hinge moment
        #  coefficients at each section
        grid = np.column_stack((flap_ratio, flap_angles, aoa_eff))
        cd0 = self.wings[wing_label]["cd0"](grid)
        cmy = self.wings[wing_label]["cmy"](grid)
        cp_min = self.wings[wing_label]["cp_min"](grid)
        ch = self.wings[wing_label]["ch"](grid)

        # Calculate cavitation number at each section
        sigma = cp_min + (static_p - pv) / dynamic_pressure

        # Calculate distribution of hinge moments at each section of the flaps
        distribution_of_hinge_moments = {}
        total_hinge_moments = {}
        for key in preprocessed_prandtl["flap angles"].keys():
            mask = preprocessed_prandtl["flap mask"] == key
            chosen_chords = chords[mask]
            chosen_ch = ch[mask]
            distribution_of_hinge_moments[key] = dynamic_pressure * chosen_chords ** 2 * chosen_ch
            total_hinge_moments[key] = distribution_of_hinge_moments[key].dot(
                hinge_integral_weights[key])

        # Calculate distributions of non-oriented forces over the whole wing
        distribution_of_lift = density * speed * circulation
        cl_eff_v2 = preprocessed_prandtl["cla"] * (np.radians(aoa_eff) - aoa_nl)
        distribution_of_lift_v2 = dynamic_pressure * chords * cl_eff_v2
        distribution_of_parasitic_drag = dynamic_pressure * chords * cd0
        distribution_of_induced_drag = distribution_of_lift * aoa_i
        distribution_of_pitch_moment = dynamic_pressure * chords ** 2 * cmy

        # project forces at each section attending to the angle of attack due to the water flow
        distribution_of_forces = np.zeros((n_points, 3))
        aoa_near = aoa_flow + aoa_i
        distribution_of_forces[:, 0] = (-distribution_of_lift * np.sin(aoa_near) +
                                        distribution_of_parasitic_drag * np.cos(aoa_near))
        if is_strut:
            distribution_of_forces[:, 1] = -(distribution_of_lift * np.cos(aoa_near) +
                                             distribution_of_parasitic_drag * np.sin(aoa_near))
        else:
            distribution_of_forces[:, 2] = (distribution_of_lift * np.cos(aoa_near) +
                                            distribution_of_parasitic_drag * np.sin(aoa_near))

        # Calculate distributions of moments
        distribution_of_moments = np.zeros((n_points, 3))
        x = preprocessed_prandtl["x"]
        distribution_of_moments[:, 0] = distribution_of_forces[:, 2] * x[:, 1]
        distribution_of_moments[:, 1] = (distribution_of_forces[:, 0] * x[:, 2] -
                                         distribution_of_forces[:, 2] * x[:, 0] +
                                         distribution_of_pitch_moment)
        distribution_of_moments[:, 2] = -distribution_of_forces[:, 0] * x[:, 1]

        # Integrate forces and moments
        total_force = wing_integral_weights.dot(distribution_of_forces)
        total_moment = wing_integral_weights.dot(distribution_of_moments)

        # Store information for further use
        solution = {"circulation coefficients": circulation_amplitudes,
                    "circulation": circulation,
                    "induced aoa": aoa_i,
                    "geometric aoa": aoa_geo,
                    "twist aoa": aoa_twist,
                    "flow aoa": aoa_flow,
                    "null lift aoa": aoa_nl,
                    "effective aoa": aoa_eff,
                    "cd0": cd0,
                    "cmy": cmy,
                    "distribution of lift": distribution_of_lift,
                    "distribution of lift v2": distribution_of_lift_v2,
                    "distribution of parasitic drag": distribution_of_parasitic_drag,
                    "distribution of induced drag": distribution_of_induced_drag,
                    "distribution of forces": distribution_of_forces,
                    "distribution of moments": distribution_of_moments,
                    "total force": total_force,
                    "total moment": total_moment,
                    "distribution of hinge moments": distribution_of_hinge_moments,
                    "total hinge moments": total_hinge_moments,
                    "cavitation number": sigma,
                    "Reynolds number": density * speed * (area / span) / viscosity
                    }

        return solution

    @staticmethod
    def _calculate_integral_weights(dx, n):
        """! Calculates the weight to perform numerical integrals of a function, i.e.,
         I = sum( f(x_i) w_i)
        @param dx: step between two consecutive values of the independent variable
        @param n: number of points in which the range is going to be discretized. If n is even, the
         weights for the Trapezoidal rule will be provided; otherwise, if n is odd, the Simpson's
         rule weights are returned. The latter provide a better approximation for the true value of
         the integral.
        @return w_i, a NumPy array of length n that contains the weights
        """
        integral_weights = dx * np.ones((n,))
        if np.mod(n, 2) == 0:
            integral_weights[0] /= 2
            integral_weights[-1] /= 2
        else:
            integral_weights /= 3
            for kk in range(1, n - 1, 2):
                integral_weights[kk] *= 4
            for kk in range(2, n - 2, 2):
                integral_weights[kk] *= 2

        return integral_weights

    @staticmethod
    def _extend_symmetrically(v, input_type):
        """ Extends an array using a point symmetry along the central element. The array must
         have an odd number of element in its first dimension. Depending on the value of the
         variable `input_type`, this method can be used to extend arbitrary first-orde NumPy
         arrays, arrays of positions of nodes (second-order array with 3 elements in the second
         dimension), or masks used to mark the nodes covered by flaps.
        @param v: NumPy array of values to ve extended symmetrically. Must have an odd number of
         elements.
        @param input_type: label that indicates which symmetry type must be applied.
         * If `vector`, the first half of the vector is point symmetrically copied into the
         second half, that is, the first half is copied in reverse order into the second. The
         central point remains un changed.
         * If `x`, the same procedure case `vector`but for arrays of two dimensions in which the
         second dimension has size 3. Elements are point-symmetrically copied along the first
         dimension and the sign of the third component of the second dimension is shifted.
         * If `flap mask`,  an array of strings has been provided and it is point-symmetrically
          extended by copying the `None` values or adding and ` S` to any string.
        @return: extended array
        """
        n_elements = len(v)
        if n_elements % 2 == 0:
            raise ValueError("Number of points in which wings are discretized must be odd")
        origin_of_symmetry = int((n_elements - 1) / 2)
        if input_type == "vector":
            for pp in range(origin_of_symmetry, n_elements):
                v[pp] = v[n_elements - 1 - pp]
        elif input_type == "x":
            for pp in range(origin_of_symmetry, n_elements):
                v[pp, :] = v[n_elements - 1 - pp, :]
                v[pp, 2] *= -1
        elif input_type == "flap mask":
            for pp in range(origin_of_symmetry, n_elements):
                if v[n_elements - 1 - pp] is not None:
                    v[pp] = v[n_elements - 1 - pp] + " S"
                else:
                    v[pp] = v[n_elements - 1 - pp]
        else:
            raise ValueError("type of input not recognized")
        return v

    @staticmethod
    def _check_wave_parameters(wave_parameters):
        """ Checks that a correct definition of the wave is being codified into the
        `wave_parameters` dictionary.
        @param wave_parameters: dictionary with the following keys
         * `distribution`: can take values `uniform` or `complete` to calculate the wave values
         at the middlepoint and extend it to all the nodes or calculate the exact value at all
         points, respectively.
         * `crest trough height [m]`: wave height
         * `wave number [m]`: wave number
         * `phase shift [rad]`: phase
         * `depth [m]`: distance from the unperturbed sea surface to the seabed.
        @return: None
        """

        if not((wave_parameters is None) or (isinstance(wave_parameters, dict))):
            raise ValueError("`wave_parameters` must be None or a dictionary")

        if wave_parameters is None:
            return

        list_of_compulsory_keys = ["distribution", "crest trough height [m]", "wave number [m]",
                                   "phase shift [rad]", "depth [m]"]
        set_of_actual_keys = set(wave_parameters.keys())

        excess_keys = set_of_actual_keys.difference(set(list_of_compulsory_keys))
        if len(excess_keys) != 0:
            raise ValueError("Key(s) {} must not be in the dictionary "
                             "`wave_parameters`".format(excess_keys))

        absent_keys = set(list_of_compulsory_keys).difference(set_of_actual_keys)
        if len(absent_keys) != 0:
            raise ValueError("Key(s) {} must be in the dictionary "
                             "`wave_parameters`".format(absent_keys))

        if ((not isinstance(wave_parameters["distribution"], str)) or
                (wave_parameters["distribution"]) not in ["uniform", "complete"]):
            raise ValueError("The value of `wave_parameters` with key 'distribution' must be "
                             "'uniform' or 'complete'")
        for key in ["crest trough height [m]", "wave number [m]", "phase shift [rad]"]:
            if not isinstance(wave_parameters[key], np.ndarray):
                raise ValueError("The value of `wave_parameters` with key '{}' must be a NumPy "
                                 "array".format(key))

        if len(wave_parameters["crest trough height [m]"].shape) != 1:
            raise ValueError("The value of `wave_parameters` with key 'crest trough height [m]' "
                             "must be an 1-dimensional NumPy array")

        n_modes = len(wave_parameters["crest trough height [m]"])

        if ((len(wave_parameters["wave number [m]"].shape) != 2) or
                (wave_parameters["wave number [m]"].shape[0] != n_modes) or
                (wave_parameters["wave number [m]"].shape[1] != 3)):
            raise ValueError("The value of `wave_parameters` with key 'wave number [m]' "
                             "must be an 2-dimensional NumPy array of shape (#modes, 3)")

        if np.linalg.norm(wave_parameters["wave number [m]"][:, 2]) > 1e-12 * n_modes:
            raise ValueError("The third (z) value of `wave_parameters` with key "
                             "'wave number [m]' must be zero for all modes")

        if ((len(wave_parameters["phase shift [rad]"].shape) != 1) or
                (wave_parameters["phase shift [rad]"].shape[0] != n_modes)):
            raise ValueError("The value of `wave_parameters` with key 'phase shift [rad]' "
                             "must be an 1-dimensional NumPy array of shape (#modes,)")

        if ((wave_parameters["depth [m]"] is not None) and
                (not isinstance(wave_parameters["depth [m]"], (int, float))) and
                (wave_parameters["depth [m]"] <= 0)):
            raise ValueError("The value of `wave_parameters` with key 'depth [m]' must be None, "
                             "an integer, or a float greater then zero")

    def _get_wave_values(self, t, x, wave_parameters, only_provide_h=False):
        """ Calculates wave properties, namely, wave height, speed, and pressure at the time and
         positions inputted by the user.
        @param t: scalar, time in seconds
        @param x: two-dimensional NumPy array with a size of 3 in the second index. Contains all
         the points where the wave properties will be evaluated.
        @param wave_parameters: dictionary with the following keys
         * `distribution`: can take values `uniform` or `complete` to calculate the wave values
         at the middlepoint and extend it to all the nodes or calculate the exact value at all
         points, respectively.
         * `crest trough height [m]`: wave height
         * `wave number [m]`: wave number
         * `phase shift [rad]`: phase
         * `depth [m]`: distance from the unperturbed sea surface to the seabed.
        @param only_provide_h: Boolean, if True only the wave height is returned; otherwise, also
         wave speed and wave pressure are calculated.
        @return: wave height, speed, and pressure at the chosen nodes. If only_provide_h is
        True, speed and pressure are None
        """

        g = 9.81
        n_points = x.shape[0]

        def extend_modal_values_to_all_point(model_values):
            return np.vstack([model_values] * n_points).T

        if wave_parameters is None:
            h = np.zeros((n_points,))
            speed = np.zeros((n_points, 3))
            p = np.zeros((n_points,))
        else:
            n_modes = len(wave_parameters["crest trough height [m]"])
            a = wave_parameters["crest trough height [m]"] / 2
            k = wave_parameters["wave number [m]"]
            phase = wave_parameters["phase shift [rad]"]
            depth = wave_parameters["depth [m]"]
            magnitude_k = np.linalg.norm(k, axis=1)
            direction_k = k / np.vstack([magnitude_k] * 3).T
            if depth is None:
                omega = np.sqrt(g * magnitude_k)
            else:
                omega = np.sqrt(g * magnitude_k * np.tanh(depth))
            a = extend_modal_values_to_all_point(a)
            omega = extend_modal_values_to_all_point(omega)
            phase = extend_modal_values_to_all_point(phase)

            argument = k @ x.T - omega * t + phase
            sin_value = np.sin(argument)
            cos_value = np.cos(argument)

            h = np.sum(a * cos_value, axis=0)

            if only_provide_h:
                speed = None
                p = None
            else:

                z = -(x[:, 2] + h)

                if depth is None:
                    u_factor = np.exp(np.outer(magnitude_k, z))
                    v_factor = u_factor.copy()
                    p_factor = u_factor.copy()
                else:
                    k_by_depth = magnitude_k * depth
                    u_factor = np.cosh(np.outer(magnitude_k, (depth + z))) / np.sinh(k_by_depth)
                    v_factor = np.sinh(np.outer(magnitude_k, (depth + z))) / np.sinh(k_by_depth)
                    p_factor = np.cosh(np.outer(magnitude_k, (depth + z))) / np.cosh(k_by_depth)
                u = a * omega * cos_value * u_factor
                v = a * omega * sin_value * v_factor
                p = np.sum(self.properties["water density [kg/m3]"] * g * a * cos_value * p_factor, axis=0)

                speed = np.zeros((n_points, 3))
                for mm in range(n_modes):
                    speed += np.outer(u[mm, :], direction_k[mm, :])
                speed[:, 2] = np.sum(v, axis=0)

        return h, speed, p

    def calculate_forces_and_moments(self, t, attitude, attitude_dot, wing_angles,
                                     flap_angles, wave_parameters=None,
                                     ground_wind_speed=np.array([0, 0, 0]),
                                     ground_water_speed=np.array([0, 0, 0]),
                                     thrust_params=None,
                                     calculate_ground_variables=False,
                                     preproc_main=None, preproc_elevator=None,
                                     preproc_strut=None):
        """Calculates forces and moments at the center of gravity.
         Hydrodynamic forces over wings and struts, aerodynamic forces over the hull, the weight,
         and the thrust of the propellers are taken into account.
        @param t: time, in seconds
        @param attitude: 1d NumPy array of length 6 with the x, y, and z positions of the CoG of
         the boat and the yaw, pitch, and roll angles.
        @param attitude_dot: 1d NumPy array of length 6 with the concatenated linear and angular
         positions of the boat.
        @param wing_angles: dictionary with the values of the geometric angles of attack (rake)
         for each lifting surface, in degrees
        @param flap_angles: dictionary that contains the values of the angles of the flaps,
         in degrees
        @param wave_parameters: as in the method `_get_wave_values`
        @param ground_wind_speed: 1d NumPy array with the three components of the wind speed
         w.r.t. the ground
        @param ground_water_speed: 1d NumPy array with the three components of the water speed
         w.r.t. the ground
        @param thrust_params: Dictionary with the type of thrust used. Key `type` is
         compulsory and can take values
          i) `constant speed`, that evaluates the required thrust to keep the boat balanced in the
          ground x-axis,
          ii) `constant thrust`, which keeps thrust constant at the value provided under the key
          "thrust [N]",
          iii) `propellers`, which uses the propellers defined in the init method and must
          contain the keys `stbd propeller` and `port propeller` related to the values of the
          rotation speed in revolutions per second.
        Cases i) and ii) might use the optional key `differential thrust` which is the amount of
         the additional thrust provided by the starboard propeller compared to the port one. If `T`
          is the total thrust of the boat, the starboard propeller provides
         `T/2 * (1 + differential_thrust)´ while the port propeller provides
         `T/2 * (1 - differential_thrust)´
        @param preproc_main: dictionary of the preprocess geometry of the main wing required by
         `prandtl_lifting_line_solve`
        @param preproc_elevator: dictionary of the preprocess geometry of the elevator required by
         `prandtl_lifting_line_solve`
        @param preproc_strut: dictionary of the preprocess geometry of the strut required by
         `prandtl_lifting_line_solve`
        @return: dictionary that contains the vectors of total forces and total moments, plus many
         partial results
        """

        if not isinstance(thrust_params, dict):
            raise TypeError("`thrust_params` is not a dictionary")
        if thrust_params.get("type", None) not in ["constant speed", "constant thrust",
                                                   "propellers"]:
            raise ValueError("Value of key `type` in `thrust_params` not recognized or key not "
                             "defined")

        linear_speed = attitude_dot[:3]
        angular_speed = attitude_dot[3:]

        ground2boat = self.get_ground2boat_matrix(attitude)
        boat2ground = ground2boat.T
        wing2boat = np.array([[-1, 0, 0],
                              [0, 1, 0],
                              [0, 0, -1]])

        def calculate_wave_properties(time, p, n_points, reference_point, wave_params,
                                      transition_time=10):
            """
            Calculate the properties of the wave at the nodes of the wing, skipping the
             calculation if no wave or uniform distribution is present. Points can be provided
             in the boat reference frame. Also, wave can be introduced slowly, multiplying its
             height by a growing-with-time factor, allowing the boat to settle to the new wave;
             this is not a physical model, but a mathematical sleight-of-hand to model the
             actual, physical situations that the boat will encounter.
            @param time: time, in seconds
            @param p: list of points, in wing frame, where the wave properties must be
             calculated
            @param n_points: number of nodes where speed must be calculated
            @param reference_point: origin of the wing reference frame from the boat frame
            @param wave_params: as in the method `_get_wave_values`
            @param transition_time: time before wave reaches its inputted height. From zero to
             this value, the intensity of the wave will be damped.
            @return: dictionary with height, speed, and pressure due to the wave
            """

            if wave_params is None:
                wave_height = np.zeros(n_points)
                wave_speed = np.zeros((n_points, 3))
                wave_pressure = np.zeros(n_points)
            else:
                # Damping factor of the wave
                if time < transition_time:
                    argument = np.pi / 2 * (2 * time / transition_time - 1)
                    if (argument + np.pi / 2) < 1e-10:
                        factor = 0
                    elif (argument - np.pi / 2) < 1e-10:
                        factor = 1
                    else:
                        factor = 1 / (1 + np.exp(-2 * np.tan(argument)))
                else:
                    factor = 1

                mod_wave_params = wave_params.copy()
                mod_wave_params["crest trough height [m]"] = (wave_params["crest trough height [m]"]
                            * factor)

                if wave_params["distribution"] == "uniform":
                    p0 = np.array([reference_point])

                    wh, ws, wp = self._get_wave_values(time, ground2boat.dot(p0.T).T, mod_wave_params)
                    wave_height = wh[0] * np.ones(n_points)
                    wave_speed = np.tile(ws[0, :], (n_points, 1))
                    wave_pressure = wp[0] * np.ones(n_points)
                elif wave_params["distribution"] == "complete":
                    p_boat = ground2boat.dot(p.T).T
                    results = self._get_wave_values(time, p_boat, mod_wave_params)
                    wave_height, wave_speed, wave_pressure = results
                else:
                    raise ValueError("The value of `wave_parameters` with key 'distribution' must "
                                     " be 'uniform' or 'complete'")

            return {"height": wave_height, "speed": wave_speed, "pressure": wave_pressure}

        # Calculate weight
        boat_weight = self.boat["mass [kg]"] * 9.81 * ground2boat.dot(np.array([0, 0, 1]))

        # Calculate main wing
        main_angles = flap_angles["main wing"]
        if preproc_main is None:
            preproc_main = self.prandtl_lifting_line_preprocess("main wing",
                                                                flap_angles=main_angles)
        main_wing_aoa = np.radians(wing_angles["main wing"])
        aoa_geo = main_wing_aoa * np.ones(self.n_points["main wing"])
        aoa_twist = np.zeros(self.n_points["main wing"])
        x = self.main_wing["reference point [m]"] + preproc_main["x"] - self.boat["CoG [m]"]

        main_wing_wave = calculate_wave_properties(t, x, self.n_points["main wing"],
                                                   self.main_wing["reference point [m]"],
                                                   wave_parameters)
        total_speed = (-ground2boat.dot(ground_water_speed) -
                       ground2boat.dot(main_wing_wave["speed"].T).T +
                       linear_speed +
                       np.cross(angular_speed, x))
        aoa_flow = total_speed[:, 2] / total_speed[:, 0]
        speed = np.average(np.sqrt(total_speed[:, 0] ** 2 + total_speed[:, 2] ** 2))
        main_wing = self.prandtl_lifting_line_solve(preproc_main, speed, aoa_geo, aoa_flow,
                                                    aoa_twist)
        x0 = self.main_wing["reference point [m]"] - self.boat["CoG [m]"]
        main_wing_force = wing2boat.dot(main_wing["total force"])
        main_wing_moment = wing2boat.dot(main_wing["total moment"]) + np.cross(x0, main_wing_force)

        # Calculate elevator
        if preproc_elevator is None:
            preproc_elevator = self.prandtl_lifting_line_preprocess("elevator",
                                                                    flap_angles={})
        aoa_geo = np.radians(wing_angles["elevator"]) * np.ones(self.n_points["elevator"])
        aoa_twist = np.zeros(self.n_points["elevator"])
        x = self.elevator["reference point [m]"] + preproc_elevator["x"] - self.boat["CoG [m]"]
        elevator_wave = calculate_wave_properties(t, x, self.n_points["elevator"],
                                                  self.elevator["reference point [m]"],
                                                  wave_parameters)
        total_speed = (-ground2boat.dot(ground_water_speed) -
                       ground2boat.dot(elevator_wave["speed"].T).T +
                       linear_speed +
                       np.cross(angular_speed, x))
        aoa_flow = total_speed[:, 2] / total_speed[:, 0]
        speed = np.average(np.sqrt(total_speed[:, 0] ** 2 + total_speed[:, 2] ** 2))
        elevator = self.prandtl_lifting_line_solve(preproc_elevator, speed, aoa_geo,
                                                   aoa_flow, aoa_twist)
        x0 = self.elevator["reference point [m]"] - self.boat["CoG [m]"]
        elevator_force = wing2boat.dot(elevator["total force"])
        elevator_moment = wing2boat.dot(elevator["total moment"]) + np.cross(x0, elevator_force)

        # Calculate wetted span of the strut
        ref_position = np.array([self.strut["reference point [m]"] - self.boat["CoG [m]"]])
        deck_position = ref_position.copy()
        deck_position[:, 2] = 0
        measures, _ = self._calculate_position_of_interface(t, deck_position, attitude,
                                                            wave_parameters, boat2ground)
        strut_span = ref_position[0, 2] - measures[0]

        # Calculate strut
        strut_angles = flap_angles["strut"]
        if preproc_strut is None:
            preproc_strut = self.prandtl_lifting_line_preprocess("strut", span=strut_span,
                                                                 flap_angles=strut_angles)
        aoa_geo = np.zeros(self.n_points["strut"])
        aoa_twist = np.zeros(self.n_points["strut"])
        x = self.strut["reference point [m]"] + preproc_strut["x"] - self.boat["CoG [m]"]
        strut_wave = calculate_wave_properties(t, x, self.n_points["strut"],
                                               self.strut["reference point [m]"],
                                               wave_parameters)
        total_speed = (-ground2boat.dot(ground_water_speed) -
                       ground2boat.dot(strut_wave["speed"].T).T +
                       linear_speed +
                       np.cross(angular_speed, x))
        aoa_flow = total_speed[:, 1] / total_speed[:, 0]
        speed = np.average(np.sqrt(total_speed[:, 0] ** 2 + total_speed[:, 1] ** 2))
        strut = self.prandtl_lifting_line_solve(preproc_strut, speed, aoa_geo, aoa_flow,
                                                aoa_twist)
        x0 = self.strut["reference point [m]"] - self.boat["CoG [m]"]
        strut_force = wing2boat.dot(strut["total force"])
        strut_moment = wing2boat.dot(strut["total moment"]) + np.cross(x0, strut_force)

        # Calculate hull
        hull_location = self.hull["center of drag [m]"] - self.boat["CoG [m]"]
        speed_vector = ground2boat.dot(ground_wind_speed)
        speed_vector -= linear_speed + np.cross(angular_speed, hull_location)
        speed_magnitude = np.linalg.norm(speed_vector)
        speed_direction = speed_vector / speed_magnitude
        hull_area = np.abs(speed_direction).dot(self.hull["areas [m2]"])
        dynamic_pressure = 0.5 * self.properties["air density [kg/m3]"] * speed_magnitude ** 2
        hull_drag = dynamic_pressure * hull_area * self.hull["CD0 [-]"]
        hull_force = hull_drag * speed_direction
        hull_moment = np.cross(hull_location, hull_force)

        # Calculate fuselage
        fuselage_location = self.fuselage["center of drag [m]"] - self.boat["CoG [m]"]
        speed_vector = ground2boat.dot(ground_water_speed)
        speed_vector -= linear_speed + np.cross(angular_speed, hull_location)
        speed_magnitude = np.linalg.norm(speed_vector)
        speed_direction = speed_vector / speed_magnitude
        fuselage_area = np.abs(speed_direction).dot(self.fuselage["areas [m2]"])
        dynamic_pressure = 0.5 * self.properties["water density [kg/m3]"] * speed_magnitude ** 2
        fuselage_drag = dynamic_pressure * fuselage_area * self.fuselage["CD0 [-]"]
        fuselage_force = fuselage_drag * speed_direction
        fuselage_moment = np.cross(fuselage_location, fuselage_force)

        # Calculate required total thrust
        total_force = (boat_weight + main_wing_force + elevator_force + strut_force + hull_force +
                       fuselage_force)

        # Calculate thrust for both propellers
        eta_prop = self.propellers["propulsive efficiency [-]"]
        thrust_direction = np.array([np.cos(main_wing_aoa), 0, -np.sin(main_wing_aoa)])

        stbd_propeller_x = np.array([self.propellers["longitudinal position [m]"],
                                     self.propellers["lateral position [m]"],
                                     self.propellers["depth [m]"]])
        stbd_speed = np.linalg.norm(linear_speed + np.cross(angular_speed, stbd_propeller_x))
        port_propeller_x = np.array([self.propellers["longitudinal position [m]"],
                                     -self.propellers["lateral position [m]"],
                                     self.propellers["depth [m]"]])
        port_speed = np.linalg.norm(linear_speed + np.cross(angular_speed, port_propeller_x))

        thrust_type = thrust_params["type"]
        if thrust_type in ["constant speed", "constant thrust"]:
            differential_thrust = thrust_params.get("differential thrust", 0)
            if thrust_type == "constant speed":
                total_thrust = -boat2ground.dot(total_force)[0] / np.cos(main_wing_aoa)
            elif thrust_type == "constant thrust":
                total_thrust = thrust_params.get("thrust [N]", None)
                if not isinstance(total_thrust, (int, float)):
                    raise ValueError("Inputted value of the thrust must be an integer or a float")
            average_propeller_force = 0.5 * total_thrust * thrust_direction
            stbd_propeller_force = average_propeller_force * (1 + differential_thrust)
            port_propeller_force = average_propeller_force * (1 - differential_thrust)
            n_stbd = self.propellers["rotational speed [rev/s]"]
            n_port = n_stbd
        elif thrust_type == "propellers":
            n_stbd = thrust_params["stbd propeller"]
            n_port = thrust_params["port propeller"]
            if "differential rotation speed" in thrust_params:
                differential_rotation_speed = thrust_params["differential rotation speed"]
                n_stbd *= (1 + differential_rotation_speed)
                n_port *= (1 - differential_rotation_speed)
            force = self.propellers["stbd propeller"].thrust(stbd_speed, n_stbd,
                                                             self.properties["water density [kg/m3]"],
                                                             check_bounds="No") # TODO: check if this is reasonable
            stbd_propeller_force = force * thrust_direction
            force = self.propellers["port propeller"].thrust(port_speed, n_port,
                                                             self.properties["water density [kg/m3]"],
                                                             check_bounds="No")
            port_propeller_force = force * thrust_direction
            total_thrust = np.linalg.norm(stbd_propeller_force + port_propeller_force)
            differential_thrust = (np.linalg.norm(stbd_propeller_force -
                                                  port_propeller_force) / total_thrust)

        stbd_motor_power = np.linalg.norm(stbd_propeller_force) * stbd_speed / eta_prop
        stbd_propeller_moment = (-stbd_motor_power * thrust_direction / (2 * np.pi * n_stbd) +
                                 np.cross(stbd_propeller_x, stbd_propeller_force))

        port_motor_power = np.linalg.norm(port_propeller_force) * port_speed / eta_prop
        port_propeller_moment = (port_motor_power * thrust_direction / (2 * np.pi * n_port) +
                                 np.cross(port_propeller_x, port_propeller_force))

        # Calculate forces and moments
        total_force += stbd_propeller_force + port_propeller_force
        total_moment = (main_wing_moment + elevator_moment + strut_moment +
                        stbd_propeller_moment + port_propeller_moment +
                        hull_moment + fuselage_moment)

        dynamics = {"attitude": attitude,
                    "derivative of attitude": attitude_dot,
                    "wing angles": wing_angles,
                    "flap angles": flap_angles,
                    "differential thrust": differential_thrust,
                    "wind speed wrt ground": ground_wind_speed,
                    "water speed wrt ground": ground_water_speed,
                    "preprocessed main wing": preproc_main,
                    "solution main wing": main_wing,
                    "preprocessed elevator": preproc_elevator,
                    "solution elevator": elevator,
                    "preprocessed strut": preproc_strut,
                    "solution strut": strut,
                    "strut span": strut_span,
                    "weight": boat_weight,
                    "main wing force": main_wing_force,
                    "main wing moment": main_wing_moment,
                    "elevator force": elevator_force,
                    "elevator moment": elevator_moment,
                    "strut wave": strut_wave,
                    "strut force": strut_force,
                    "strut moment": strut_moment,
                    "hull force": hull_force,
                    "hull moment": hull_moment,
                    "fuselage force": fuselage_force,
                    "fuselage moment": fuselage_moment,
                    "total thrust": total_thrust,
                    "stbd propeller force": stbd_propeller_force,
                    "stbd propeller moment": stbd_propeller_moment,
                    "stbd motor power": stbd_motor_power,
                    "stbd propeller speed": n_stbd,
                    "port propeller force": port_propeller_force,
                    "port propeller moment": port_propeller_moment,
                    "port motor power": port_motor_power,
                    "port propeller speed": n_port,
                    "total force": total_force,
                    "total moment": total_moment}

        # Transform forces and moments from boat to ground frame
        if calculate_ground_variables:
            list_of_keys = ["weight", "main wing force", "main wing moment", "elevator force",
                            "elevator moment", "strut force", "strut moment", "hull force",
                            "hull moment", "fuselage force", "fuselage moment",
                            "stbd propeller force", "stbd propeller moment",
                            "port propeller force", "port propeller moment", "total force",
                            "total moment"]
            for key in list_of_keys:
                dynamics["ground " + key] = boat2ground.dot(dynamics[key])

        return dynamics

    @staticmethod
    def get_ground2boat_matrix(attitude):
        """ Calculates the transformation matrix from the ground frame to the boat frame
        @param attitude: vector of the state of the boat
        @return: the matrix that transform any vector in ground co-ordinates to boat co-ordinates
        """
        # Assign aliases
        yaw = attitude[3]
        pitch = attitude[4]
        roll = attitude[5]

        ground2boat = [[np.cos(yaw) * np.cos(pitch),
                        np.sin(yaw) * np.cos(pitch),
                        -np.sin(pitch)],
                       [-np.sin(yaw) * np.cos(roll) + np.sin(roll) * np.sin(pitch) * np.cos(yaw),
                        np.sin(yaw) * np.sin(roll) * np.sin(pitch) + np.cos(yaw) * np.cos(roll),
                        np.sin(roll) * np.cos(pitch)],
                       [np.sin(yaw) * np.sin(roll) + np.sin(pitch) * np.cos(yaw) * np.cos(roll),
                        np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.sin(roll) * np.cos(yaw),
                        np.cos(roll) * np.cos(pitch)]]

        return np.array(ground2boat)

    def get_boat2ground_matrix(self, attitude):
        """ Calculates the transformation matrix from the boat frame to the ground frame
        @param attitude: vector of the state of the boat
        @return: the matrix that transform any vector in boat co-ordinates to ground co-ordinates
        """
        ground2boat = self.get_ground2boat_matrix(attitude)
        return ground2boat.T

    @staticmethod
    def angular_speed_from_euler_to_boat(attitude, euler_omega):
        """ Transform from angular speed as derivatives of Euler angles to the angular speed
        vector in the boat frame
        @param attitude: vector of the state of the boat
        @param euler_omega: angular speed as the derivatives of the Euler angles w.r.t. time
        @return: angular speed vector in the boat frame
        """
        # Assign aliases
        pitch = attitude[4]
        roll = attitude[5]
        yaw_dot = euler_omega[0]
        pitch_dot = euler_omega[1]
        roll_dot = euler_omega[2]
        # Transform from angular speed as derivatives of Euler angles to the angular speed
        #  vector in the boat frame
        p = roll_dot - yaw_dot * np.sin(pitch)
        q = pitch_dot * np.cos(roll) + yaw_dot * np.cos(pitch) * np.sin(roll)
        r = -pitch_dot * np.sin(roll) + yaw_dot * np.cos(pitch) * np.cos(roll)

        return np.array([p, q, r])

    @staticmethod
    def angular_speed_from_boat_to_euler(attitude, attitude_dot):
        """ Transform from the angular speed vector in the boat frame to angular speed as
         derivatives of Euler angles
        @param attitude: vector of the state of the boat
        @param attitude_dot: angular speed vector in the boat frame
        @return: angular speed as the derivatives of the Euler angles w.r.t. time
        """
        # Assign aliases
        pitch = attitude[4]
        roll = attitude[5]
        p = attitude_dot[3]
        q = attitude_dot[4]
        r = attitude_dot[5]
        # Transform from angular speed as derivatives of Euler angles to the angular speed
        #  vector in the boat frame
        roll_dot = p + (q * np.sin(roll) + r * np.cos(roll)) * np.tan(pitch)
        pitch_dot = q * np.cos(roll) - r * np.sin(roll)
        yaw_dot = (q * np.sin(roll) + r * np.cos(roll)) / np.cos(pitch)

        return np.array([yaw_dot, pitch_dot, roll_dot])
    
    @staticmethod
    def plot_distributions_along_wings(list_of_plots, dynamics, wings_to_plot):
        """ Plots the distributions of the variables returned by the hydrodynamic solver as a
         function of the span. It can plot, e.g., the lift, the circulation, the parasitic drag,
         or the pitch moment. Each subfigure can contain several plot to compare several
         variables or to compare the same variable for several lifting surfaces.
        @param list_of_plots: list of lists that contain what must be plotted. The first level
         of the list indicates the row, and the second level the column. Each of the elements of
         the second level list must be one of the keys of the `dynamics` dictionary thet is
         returned by `prandtl_lifting_line_solve`
        @param dynamics: dictionary formatted as the output of ´calculate_forces_and_moments´
        @param wings_to_plot: list of the wings for which the variables must be plotted. It can
         take the values ´main wing´, ´elevator´, ´strut´, (to any of those lifting surfaces),
         ´horizontal´ (to plot both main wing and elevator), or ´all´ (to plot the three
         surfaces)
        @return: None
        """

        # Check that input is correctly formatted
        error_msg = "`list_of_plots` must be a list of lists than contains strings"
        if not isinstance(list_of_plots, list):
            raise ValueError(error_msg)
        n_rows = len(list_of_plots)
        n_columns = -1
        for list_of_strings in list_of_plots:
            if not isinstance(list_of_strings, list):
                raise ValueError(error_msg)
            n_columns = max([n_columns, len(list_of_strings)])
            if not np.all([isinstance(elem, str) for elem in list_of_strings]):
                raise ValueError(error_msg)

        # Check that wings exist
        if wings_to_plot not in ["horizontal", "strut", "main wing", "elevator", "all"]:
            raise ValueError("variable `wings_to_plot` have a value that is not recognized")
        elif wings_to_plot == "horizontal":
            wing_list = ["main wing", "elevator"]
        elif wings_to_plot == "all":
            wing_list = ["main wing", "elevator", "strut"]
        else:
            wing_list = [wings_to_plot]

        # Define list of colors to use
        colors = ["teal", "turquoise", "orchid", "burlywood", "coral", "fuchsia", "goldenrod",
                  "darkcyan", "olive", "magenta"]

        # Lists of variables to be plotted
        preproc_prandtl_labels = ['theta', 'span-wise x', 'x', 'chords', 'flap mask', 'flap ratio',
                                  'cla', 'aoa nl', 'symmetry mask']
        solution_labels = ['circulation', 'induced aoa', 'geometric aoa', 'twist aoa', 'flow aoa',
                           'null lift aoa', 'effective aoa', 'cd0', 'cmy', 'distribution of lift',
                           'distribution of lift v2',
                           'distribution of parasitic drag', 'distribution of induced drag',
                           'distribution of forces', 'distribution of moments',
                           'distribution of hinge moments', 'cavitation number']

        # Create figure
        fig = make_subplots(rows=n_rows, cols=n_columns)
        # Loop over plots and
        for pp, list_of_strings in enumerate(list_of_plots):
            for qq, plot_label in enumerate(list_of_strings):

                for rr, wing_label in enumerate(wing_list):

                    preproc_prandtl = dynamics["preprocessed {}".format(wing_label)]
                    solution = dynamics["solution {}".format(wing_label)]

                    symmetry_mask = np.logical_not(preproc_prandtl["symmetry mask"])

                    if wing_label != "strut":
                        x = preproc_prandtl["x"][:, 1]
                    else:
                        x = preproc_prandtl["x"][symmetry_mask, 2]

                    if plot_label in preproc_prandtl_labels:
                        y = preproc_prandtl[plot_label][symmetry_mask]
                    elif plot_label in solution_labels:
                        y = solution[plot_label][symmetry_mask]
                    else:
                        raise ValueError("the label `{}` is not recognized".format(plot_label))

                    fig.add_trace(
                        go.Scatter(
                            visible=True,
                            line=dict(color=colors[rr], width=2),
                            name="{} - {}".format(wing_label, plot_label),
                            x=x,
                            y=y),
                        row=pp+1, col=qq+1)
                    fig.update_xaxes(title_text="y [m]", row=pp+1, col=qq+1)
                    fig.update_yaxes(title_text=plot_label, row=pp+1, col=qq+1)

        fig.show()

    def sketch_me(self, n_points=101, n_sections=5):
        """ Plots the set of wings and struts. Each lifting surface is sketched by a
         wireframe-like elevation, plan, section, and 3d views. Lines for the leading and
         trailing edge are shown, also the locus of all the highest and lowest points of each
         section is plotted. Additionally, the contours of the hydrofoil are shown at a
         user-defined number of equistpaced stations.
        @param n_points: number of points in which each wing is discretized
        @param n_sections: number of sections (contours of the hydrofoil) plotted for each wing
        @return: None
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, projection='3d')
        for label, wing in self.wings.items():

            s_bounds = self.dict_of_designs[label]['s bounds [-]']

            # Get index of each of the positions to be plotted
            ii_le = np.argmin(wing["unit chord foil"][0, :])
            ii_te = np.argmax(wing["unit chord foil"][0, :])
            ii_ps = np.argmin(wing["unit chord foil"][2, :])
            ii_ss = np.argmax(wing["unit chord foil"][2, :])
            # Initialize array of lines to be plotted
            le = np.zeros((3, n_points))
            te = np.zeros((3, n_points))
            ps = np.zeros((3, n_points))
            ss = np.zeros((3, n_points))
            # Calculate contours of the foils at each point
            contours_2d = []
            for s in np.linspace(s_bounds[0], s_bounds[1], n_points):
                x = self.dict_of_designs[label]['quarter-chord [m]'](s).reshape(3, -1).squeeze()
                x[1] = -x[1]
                x[2] = -x[2]
                c = self.dict_of_designs[label]['chord [m]'](s)
                y = (self.dict_of_designs[label]['reference point [m]'] + x).reshape(3, -1)\
                    + c * wing['unit chord foil']
                contours_2d.append(y)

            # Extract coordinates of each point
            for ii, foil in enumerate(contours_2d):
                le[:, ii] = foil[:, ii_le]
                te[:, ii] = foil[:, ii_te]
                ps[:, ii] = foil[:, ii_ps]
                ss[:, ii] = foil[:, ii_ss]

            contours_3d = []
            for s in np.linspace(s_bounds[0], s_bounds[1], n_sections):
                x = self.dict_of_designs[label]['quarter-chord [m]'](s).reshape(3, -1).squeeze()
                x[1] = -x[1]
                x[2] = -x[2]
                c = self.dict_of_designs[label]['chord [m]'](s)
                y = (self.dict_of_designs[label]['reference point [m]'] + x).reshape(3, -1)\
                    + c * wing['unit chord foil']
                contours_3d.append(y)

            # Plot results
            # Elevation view
            for foil in contours_3d:
                ax1.plot(foil[1, :], foil[2, :])
            ax1.plot(le[1, :], le[2, :])
            ax1.plot(te[1, :], te[2, :])
            ax1.plot(ps[1, :], ps[2, :])
            ax1.plot(ss[1, :], ss[2, :])
            ax1.set_xlabel('Y [m]')
            ax1.set_ylabel('Z [m]')
            ax1.grid(True, which='both')
            ax1.axis('equal')
            # Section view
            for foil in contours_3d:
                ax2.plot(foil[0, :], foil[2, :])
            ax2.plot(le[0, :], le[2, :])
            ax2.plot(te[0, :], te[2, :])
            ax2.plot(ps[0, :], ps[2, :])
            ax2.plot(ss[0, :], ss[2, :])
            ax2.set_xlabel('X [m]')
            ax2.set_ylabel('Z [m]')
            ax2.grid(True, which='both')
            ax2.axis('equal')
            # Plan view
            for foil in contours_3d:
                ax3.plot(foil[1, :], foil[0, :])
            ax3.plot(le[1, :], le[0, :])
            ax3.plot(te[1, :], te[0, :])
            ax3.plot(ps[1, :], ps[0, :])
            ax3.plot(ss[1, :], ss[0, :])
            ax3.set_xlabel('Y [m]')
            ax3.set_ylabel('X [m]')
            ax3.grid(True, which='both')
            ax3.axis('equal')
            # 3d interactive view
            for foil in contours_3d:
                ax4.plot(foil[0, :], foil[1, :], foil[2, :], color='k')
            ax4.plot(le[0, :], le[1, :], le[2, :])
            ax4.plot(te[0, :], te[1, :], te[2, :])
            ax4.plot(ps[0, :], ps[1, :], ps[2, :])
            ax4.plot(ss[0, :], ss[1, :], ss[2, :])
            ax4.set_xlabel('X [m]')
            ax4.set_ylabel('Y [m]')
            ax4.set_zlabel('Z [m]')
            ax4.grid(True, which='both')

        matplotlib_extended.set_axes_equal(ax4)
        fig.suptitle("Sketch of the boat")
        plt.show()

    @staticmethod
    def _ittc_water_properties(t=15, sa=35.16504):
        """ Calculate the water properties for a given temperature and salinity
        Correlation obtained from
          * ITTC – Recommended Procedures - Fresh Water and Seawater Properties
          * Thermophysical properties of seawater: a review of existing correlations and data
            Mostafa H. Sharqawy, John H. Lienhard V & Syed M. Zubair, 2010
            https://doi.org/10.5004/dwt.2010.1079
        @param t: temperature of the water in Celsius degrees
        @param sa: salinity of the water in g / kg
        @return: values of the density (kg /m3), viscosity (Pa s), and vapour pressure (Pa)
        """

        # Check the bounds of the correlations used
        if not ((-2 <= t <= 40) and (0 <= sa <= 42)):
            raise ValueError("temperature or salinity are out of the bounds of the water "
                             "correlation")

        # Calculate water density in kg / m3
        a = 0.824493 - 4.0899e-3 * t + 7.6438e-5 * t ** 2 - 8.2467e-7 * t ** 3 + 5.3875e-9 * t ** 4
        b = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2
        c = 4.8314e-4
        rho_fw = (999.842594 + 6.793952e-2 * t - 9.09529e-3 * t ** 2 + 1.001684e-4 * t ** 3 -
                  1.120083e-6 * t ** 4 + 6.536336e-9 * t ** 5)

        rho_sw = rho_fw + a * sa + b * sa ** (3/2) + c * sa ** 2

        # temperature in Celsius degrees
        temperature = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                       30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                       40]
        # viscosity in Pa s
        viscosity_fw = [0.001306, 0.001269, 0.001234, 0.001200, 0.001168,
                        0.001138, 0.001108, 0.001080, 0.001053, 0.001027,
                        0.001002, 0.000978, 0.000954, 0.000932, 0.000911,
                        0.000890, 0.000870, 0.000851, 0.000832, 0.000814,
                        0.000797, 0.000781, 0.000764, 0.000749, 0.000734,
                        0.000719, 0.000705, 0.000691, 0.000678, 0.000665,
                        0.000653]
        mu_fw_vs_t = interp1d(temperature, viscosity_fw, kind='cubic', bounds_error=True)
        mu_fw = mu_fw_vs_t(t)

        # pressure in kPa
        vapour_pressure_fw = [1.2282, 1.3130, 1.4028, 1.4981, 1.5990,
                              1.7058, 1.8188, 1.9384, 2.0647, 2.1983,
                              2.3393, 2.4882, 2.6453, 2.8111, 2.9858,
                              3.1699, 3.3639, 3.5681, 3.7831, 4.0092,
                              4.2470, 4.4969, 4.7596, 5.0354, 5.3251,
                              5.6290, 5.9479, 6.2823, 6.6328, 7.0002,
                              7.3849]
        pv_fw_vs_t = interp1d(temperature, vapour_pressure_fw, kind='cubic', bounds_error=True)
        pv_fw = pv_fw_vs_t(t)

        # Viscosity in Pa s
        a = 1.474e-3 + 1.5e-5 * t - 3.927e-8 * t ** 2
        b = 1.073e-5 - 8.5e-8 * t + 2.230e-10 * t ** 2
        mu_sw = mu_fw * (1 + a * sa + b * sa**2)

        pv_sw = pv_fw / (1 + 0.57357 * (sa / (1000 - sa)))

        return rho_sw, mu_sw, pv_sw * 1e3

    # TODO: Check if it is deprecated
    def find_straight_movement_solution(self, attitude, attitude_dot, max_error=1e-2,
                                        initial_wing_angles=(0, 0), step=1.0,
                                        fixed_main_wing_aoa=None,
                                        ground_water_speed=np.array([0, 0, 0]),
                                        ground_wind_speed=np.array([0, 0, 0])):
        """ Method to find the values of the angles of attack in which the longitudinal
         equilibrium can be met, i.e., the sum of z-forces and y-moments must be zero. All the
         other values are already null (e.g. no lateral velocity, nor wind or water currents).
        @param attitude: vector of the state of the boat
        @param attitude_dot: angular speed vector in the boat frame
        @param max_error: value of the maximum residual allowed in the sum of z-forces and
         y-moments when solving. Alternatively, a tuple of 2 values can be inputted: the first
         will be used for the sum of z-forces and the second for the sum of y-moments
        @param initial_wing_angles: a 2-tuple of the values of the main wing and the elevator, in
         degrees, used to initiate the iterative solver. If `fixed_main_wing_aoa` is not None,
         the first value is not the AoA of the main wing, which is fixed and its inputted in the
         other variable; instead it is the initial value of the flap angles.
        @param step: variation in the angle of attack or the flap from one step of the iterative
         solving method to the next. This values is reduced as the solution is converged.
        @param fixed_main_wing_aoa: if None, the whole main wing is moved to find the
         equilibrium. If a float is provided, it will be set as the rake of the main wing and
         the flaps will be moved to modify the lift and meet zero-sum of forces and moments.
        @param ground_water_speed:
        @param ground_wind_speed:
        @return: the same dictionary returned by `self.calculate_forces_and_moments` calculated
         at the equilibrium point.
        """
        # Perform some preliminary checks over the inputs
        if isinstance(max_error, float):
            max_error = (max_error, max_error)
        elif isinstance(max_error, (list, tuple, np.ndarray)) and (len(max_error) != 2):
            raise ValueError('`max_error` must have a length of 2')
        else:
            raise ValueError('`max_error` must be a float, a tuple, a list, or a NumPy array')
    
        if (fixed_main_wing_aoa is not None) and (not isinstance(fixed_main_wing_aoa, float)):
            raise ValueError('`fixed_main_wing_aoa` must be None or a float')
    
        # Set initial values
        thrust_params = {"type": "constant speed"}
        if fixed_main_wing_aoa is None:
            wing_angles = {"main wing": initial_wing_angles[0], "elevator": initial_wing_angles[1]}
            flap_angles = {"main wing": {"stbd": 0, "port": 0}, "strut": {"upper": 0, "lower": 0}}
        else:
            wing_angles = {"main wing": fixed_main_wing_aoa, "elevator": initial_wing_angles[1]}
            flap_angles = {"main wing": {"stbd": initial_wing_angles[0],
                                         "port": initial_wing_angles[0]},
                           "strut": {"upper": 0, "lower": 0}}
    
        it = 0
        h1 = []
        h2 = []
        # Initial values
        solution = {"preprocessed main wing": None,
                    "preprocessed elevator": None,
                    "preprocessed strut": None}
        while True:
            if fixed_main_wing_aoa is None:
                h1.append(wing_angles["main wing"])
            else:
                h1.append(flap_angles["main wing"]["stbd"])
            h2.append(wing_angles["elevator"])
    
            if fixed_main_wing_aoa is not None:
                preproc_main = None
            else:
                preproc_main = solution["preprocessed main wing"]
    
            preproc_elevator = solution["preprocessed elevator"]
            preproc_strut = solution["preprocessed strut"]
    
            solution = self.calculate_forces_and_moments(t=0, attitude=attitude,
                                                         attitude_dot=attitude_dot,
                                                         wing_angles=wing_angles,
                                                         flap_angles=flap_angles,
                                                         thrust_params=thrust_params,
                                                         ground_wind_speed=ground_wind_speed,
                                                         ground_water_speed=ground_water_speed,
                                                         preproc_main=preproc_main,
                                                         preproc_elevator=preproc_elevator,
                                                         preproc_strut=preproc_strut,
                                                         calculate_ground_variables=True)
    
            sum_fz = solution["ground total force"][2]
            sum_my = solution["ground total moment"][1]
    
            if fixed_main_wing_aoa is None:
                if (sum_fz < 0) and (sum_my > 0):
                    wing_angles["main wing"] -= step
                elif (sum_fz < 0) and (sum_my < 0):
                    wing_angles["elevator"] -= step
                elif (sum_fz > 0) and (sum_my > 0):
                    wing_angles["elevator"] += step
                elif (sum_fz > 0) and (sum_my < 0):
                    wing_angles["main wing"] += step
            else:
                if (sum_fz < 0) and (sum_my > 0):
                    flap_angles["main wing"]["stbd"] -= step
                    flap_angles["main wing"]["port"] -= step
                elif (sum_fz < 0) and (sum_my < 0):
                    wing_angles["elevator"] -= step
                elif (sum_fz > 0) and (sum_my > 0):
                    wing_angles["elevator"] += step
                elif (sum_fz > 0) and (sum_my < 0):
                    flap_angles["main wing"]["stbd"] += step
                    flap_angles["main wing"]["port"] += step
    
            it += 1
            if len(h1) >= 4:
                if (np.ptp(h1[-4:]) <= 1.1 * step) and (np.ptp(h2[-4:]) <= 1.1 * step):
                    step /= 2
                elif (np.ptp(h1[-4:]) > 2 * step) and (np.ptp(h2[-4:]) > 2 * step):
                    step *= 2
    
            if (np.abs(sum_fz) < max_error[0]) and (np.abs(sum_my) < max_error[1]):
                break
    
            if it % 10 == 0:
                print(it, step, (wing_angles["main wing"], wing_angles["elevator"]),
                      (sum_fz, sum_my))
    
        return solution

    def find_stationary_solution(self, initial_state_vector, initial_input_vector,
                                 equations_to_solve, variables_to_find,
                                 ground_wind_speed=None, ground_water_speed=None,
                                 do_not_recalculate_preproccesed_prantl=False, max_it=100,
                                 input_limits_dict: dict={}):
        """ Method to find the values of any variable chosen to set an stationary state. The
         user can define which equations must be verified and which degrees of freedom are to be
         solved for. Solution is achieved thanks to a Newton-Raphson method.
        @param initial_state_vector: 1d, 12 elements NumPy array with the state of the boat,
         i.e., the position (x, y, z), the attitude (yaw, pitch, roll), and the linear and
         angular speeds in the boat reference frame
        @param initial_input_vector: initial values of the variables to solve the equations for.
         The nearest they are to the solution the faster the convergence will be.
        @param equations_to_solve: list of equations that must be solved. It can contain any of
         the values in the list ´self.equation_labels´
        @param variables_to_find: list of the variables to solve the equations for. It can
         contain any of the values contained in ´self.state_labels´ or ´self.inputs_for_this_case´
        @param ground_wind_speed: 1d NumPy array with the three components of the wind speed
         w.r.t. the ground
        @param ground_water_speed: 1d NumPy array with the three components of the water speed
         w.r.t. the ground
        @param do_not_recalculate_preproccesed_prantl: if True, preprocessing of evey wing is
         skipped after the initial time. If False the preprocessing is repeated every iteration.
        @return: solution values of the state and input vectors, values of the residuals,
         and number of iterations required to converge
        """

        equation_index = []
        variable_index = []
        for equation in equations_to_solve:
            equation_index.append(self.equation_labels.index(equation))

        linearize_ground_equations = any([equation.startswith("ground")
                                          for equation in equations_to_solve])

        variable_labels = self.state_labels + self.inputs_for_this_case
        for variable in variables_to_find:
            variable_index.append(variable_labels.index(variable))

        it = 0
        state_vector = initial_state_vector.copy()
        input_vector = initial_input_vector.copy()
        while it < max_it:
            it += 1
            f0, a, b = self.linearize(state_vector=state_vector, input_vector=input_vector,
                                      ground_wind_speed=ground_wind_speed,
                                      ground_water_speed=ground_water_speed,
                                      do_not_recalculate_preproccesed_prantl=do_not_recalculate_preproccesed_prantl,
                                      linearize_ground_equations=linearize_ground_equations)

            c = np.hstack([a, b])
            mat = c[equation_index, :][:, variable_index]
            vec = f0[equation_index]
            delta = np.linalg.solve(mat, -vec)

            if (np.linalg.norm(vec) < 1e-6 * len(vec)) and (
                    np.linalg.norm(delta) < 1e-6 * len(delta)):
                break
            else:
                for qq, index in enumerate(variable_index):
                    if index < 12:
                        state_vector[index] += delta[qq]
                    else:
                        input_vector[index - 12] += delta[qq]
                # print(f"\ndelta: {delta}")
                # print(f"state_vector: {state_vector}")
                # print(f"input_vector: {input_vector}")
                
        if pyairy.Control.clip_inputs(input_vector, [input_limits_dict.get(label, None) for label in self.inputs_for_this_case])[1]:
            raise ValueError(f"Inputs are out of bounds: {input_vector}")
                        
        return state_vector, input_vector, f0, it

    @staticmethod
    def get_state(attitude, attitude_dot):
        """
        Concatenates the position, attitude, and speeds of the boat into a single, state vector
        @param attitude: 1d, 6 elements, NumPy array with the position of the CoG (x, y,z) and the
         attitude of the boat (yaw, pitch, roll)
        @param attitude_dot: 1d, 6 elements, NumPy array with the linear and angular speeds
         concatenated
        @return: state of the boat
        """
        return np.concatenate((attitude, attitude_dot))

    @staticmethod
    def split_state(state):
        """
        Split the state vector into the attitude of the boat (position of CoG + Euler angles) 
         and the speeds (both linear and angular) 
        @param state: 1d, 12 elements, NumPy array with the state of the boat, i.e., the position
         (x, y, z), the attitude (yaw, pitch, roll), and the linear and angular speeds in the 
         boat reference frame. 
        @return: 1d, 6-elements, NumPy arrays of the attitude and the speeds of the boat
        """
        return state[:6], state[6:]

    def set_inputs_for_this_case(self, inputs_for_this_case):
        """
        Sets the values of the variables that will be the inputs, the variables that can be
         changed to control the state of the boat
        @param inputs_for_this_case: list of strings, the inputs
        @return: None
        """
        if not isinstance(inputs_for_this_case, list):
            raise ValueError("`inputs_for_this_case` must be a list of labels")

        list_of_labels = ["main wing", "elevator", "flap stbd", "flap port", "flap symmetric",
                          "flap antisymmetric", "rudder", "whole rudder", "total thrust", "differential thrust", 
                          "stbd rotation speed", "port rotation speed", "propellers rotation speed", 
                          "average rotation speed", "differential rotation speed"]
        for label in inputs_for_this_case:
            if label not in list_of_labels:
                raise ValueError("label of input not recognized:", label)

        if (("stbd rotation speed" in inputs_for_this_case) !=
                ("port rotation speed" in inputs_for_this_case)):
            raise ValueError ("Both `stbd rotation speed` and `port rotation speed` must be "
                              "present or not present in `inputs_for_this_case`")
            
        if (("differential rotation speed" in inputs_for_this_case) and not
                ("average rotation speed" in inputs_for_this_case)):
            raise ValueError ("'differential rotation speed' can't be used without 'average rotation speed'", inputs_for_this_case)
        
        self.inputs_for_this_case = inputs_for_this_case
        
        
    def calculate_state_dot(self, t, state_vector, input_vector, kwargs):
        """ Calculates the derivative w.r.t. time of the state vector
        @param t: time, in seconds
        @param state_vector: 1d, 12 elements NumPy array with the state of the boat, i.e., the
         position (x, y, z), the attitude (yaw, pitch, roll), and the linear and angular speeds
         in the boat reference frame
        @param input_vector: initial values of the variables to solve the equations for. The
         nearest they are to the solution the faster the convergence will be.
        @param kwargs: dictionary with the values of "ground wind speed", "ground water speed",
         "wave parameters", "differential thrust", and "calculate_ground_state_dot". These names
         are the keys of the dictionary.
        @return: derivative of the state vector and the ´dynamics´ dictionary as outputted from
         ´calculate_forces_and_moments´
        """
        ground_wind_speed = kwargs.get("ground wind speed", np.array([0, 0, 0]))
        if ground_wind_speed is None:
            ground_wind_speed = np.array([0, 0, 0])
        ground_water_speed = kwargs.get("ground water speed", np.array([0, 0, 0]))
        if ground_water_speed is None:
            ground_water_speed = np.array([0, 0, 0])
        wave_parameters = kwargs.get("wave parameters", None)
        if wave_parameters is None:
            wave_parameters = {"distribution": "uniform",
                               "crest trough height [m]": np.array([0.0]),
                               "wave number [m]": np.array([[1, 0, 0]]),
                               "phase shift [rad]": np.array([0]),
                               "depth [m]": None}
        preprocessed_main_wing = kwargs.get("preprocessed main wing", None)
        preprocessed_elevator = kwargs.get("preprocessed elevator", None)
        preprocessed_strut = kwargs.get("preprocessed strut", None)

        calculate_ground_state_dot = kwargs.get("calculate ground state dot", False)

        # assign aliases for the sake of clarity of the following expressions
        yaw = state_vector[3]
        pitch = state_vector[4]
        roll = state_vector[5]
        u = state_vector[6]
        v = state_vector[7]
        w = state_vector[8]
        p = state_vector[9]
        q = state_vector[10]
        r = state_vector[11]

        if len(input_vector) != len(self.inputs_for_this_case):
            raise ValueError("The lists 'input_vector' and `self.inputs_for_this_case` must have equal lengths, " +
                             "now they are {} and {}".format(len(input_vector), len(self.inputs_for_this_case)))

        wing_angles = {"main wing": 0, "elevator": 0}
        flap_angles = {"main wing": {"stbd": 0, "port": 0}, "strut": {"upper": 0, "lower": 0}}
        thrust_params = {}
        input_labels = self.inputs_for_this_case + list(self.fixed_inputs.keys())
        input_vector = np.concatenate((input_vector, list(self.fixed_inputs.values())))
        for pp, label in enumerate(input_labels):
            if label == "main wing":
                wing_angles["main wing"] = input_vector[pp]
            elif label == "elevator":
                wing_angles["elevator"] = input_vector[pp]
            elif label == "flap stbd":
                flap_angles["main wing"]["stbd"] = input_vector[pp]
            elif label == "flap port":
                flap_angles["main wing"]["port"] = input_vector[pp]
            elif label == "flap symmetric":
                flap_angles["main wing"]["stbd"] = input_vector[pp]
                flap_angles["main wing"]["port"] = input_vector[pp]
            elif label == "flap antisymmetric":
                flap_angles["main wing"]["stbd"] = input_vector[pp]
                flap_angles["main wing"]["port"] = -input_vector[pp]
            elif label == "half rudder":
                flap_angles["strut"]["lower"] = input_vector[pp]
            elif label == "rudder" or label == "whole rudder":
                flap_angles["strut"]["lower"] = input_vector[pp]
                flap_angles["strut"]["upper"] = input_vector[pp]
            elif label == "total thrust":
                thrust_params["type"] = "constant thrust"
                thrust_params["thrust [N]"] = input_vector[pp]
            elif label == "differential thrust":
                thrust_params["differential thrust"] = input_vector[pp]
            elif label == "stbd rotation speed":
                thrust_params["type"] = "propellers"
                thrust_params["stbd propeller"] = input_vector[pp]
            elif label == "port rotation speed":
                thrust_params["type"] = "propellers"
                thrust_params["port propeller"] = input_vector[pp]
            elif label == "propellers rotation speed" or label == "average rotation speed":
                thrust_params["type"] = "propellers"
                thrust_params["stbd propeller"] = input_vector[pp]
                thrust_params["port propeller"] = input_vector[pp]
            elif label == "differential rotation speed":
                thrust_params["type"] = "propellers"
                thrust_params["differential rotation speed"] = input_vector[pp]
            else:
                raise ValueError("label of input not recognized")

        if thrust_params.get("type", None) is None:
            thrust_params["type"] = "constant speed"

        attitude, attitude_dot = self.split_state(state_vector)
        dynamics = self.calculate_forces_and_moments(t, attitude, attitude_dot, wing_angles,
                                                     flap_angles, thrust_params=thrust_params,
                                                     wave_parameters=wave_parameters,
                                                     ground_wind_speed=ground_wind_speed,
                                                     ground_water_speed=ground_water_speed,
                                                     preproc_main=preprocessed_main_wing,
                                                     preproc_elevator=preprocessed_elevator,
                                                     preproc_strut=preprocessed_strut)
        forces = dynamics["total force"]
        moments = dynamics["total moment"]

        linear_speed = np.array([u, v, w])
        angular_speed = np.array([p, q, r])
        inertial_force = np.cross(angular_speed, linear_speed)
        inertial_moment = np.cross(angular_speed, self.inertia_matrix.dot(angular_speed))

        if not calculate_ground_state_dot:
            state_dot = np.zeros(12)
        else:
            state_dot = np.zeros(18)
        # Kinematic laws of the COG
        state_dot[0] = (np.cos(pitch) * np.cos(yaw) * u +
                        (np.sin(roll) * np.sin(pitch) * np.cos(yaw) -
                         np.cos(roll) * np.sin(yaw)) * v +
                        (np.cos(roll) * np.sin(pitch) * np.cos(yaw) +
                         np.sin(roll) * np.sin(yaw)) * w)
        state_dot[1] = (np.cos(pitch) * np.sin(yaw) * u +
                        (np.sin(roll) * np.sin(pitch) * np.sin(yaw) +
                         np.cos(roll) * np.cos(yaw)) * v +
                        (np.cos(roll) * np.sin(pitch) * np.sin(yaw) -
                         np.sin(roll) * np.cos(yaw)) * w)
        state_dot[2] = (-np.sin(pitch) * u +
                        np.sin(roll) * np.cos(pitch) * v +
                        np.cos(roll) * np.cos(pitch) * w)
        # Evolution of Euler angles
        angular_acceleration = self.angular_speed_from_boat_to_euler(attitude, attitude_dot)
        state_dot[3] = angular_acceleration[0]
        state_dot[4] = angular_acceleration[1]
        state_dot[5] = angular_acceleration[2]
        state_dot[6:9] = -inertial_force + forces / self.boat["mass [kg]"]
        state_dot[9:12] = np.linalg.solve(self.inertia_matrix, -inertial_moment + moments)

        if calculate_ground_state_dot:
            boat2ground = self.get_boat2ground_matrix(attitude)

            state_dot[12:15] = boat2ground.dot(state_dot[6:9])
            state_dot[15:18] = boat2ground.dot(state_dot[9:12])

        return state_dot, dynamics

    def linearize(self, state_vector, input_vector, ground_wind_speed=None,
                  ground_water_speed=None, t=0, do_not_recalculate_preproccesed_prantl=False,
                  linearize_ground_equations=False):
        """ Find the linear expansion of the governing equations at the chosen point (defined by
         the state and input vectors)
        @param state_vector: 1d, 12 elements NumPy array with the state of the boat, i.e., the
         position (x, y, z), the attitude (yaw, pitch, roll), and the linear and angular speeds
         in the boat reference frame
        @param input_vector: values of the variables that can be modified to control the boat.
        @param ground_wind_speed: 1d NumPy array with the three components of the wind speed
         w.r.t. the ground
        @param ground_water_speed: 1d NumPy array with the three components of the water speed
         w.r.t. the ground
        @param t: time, in seconds.
        @param do_not_recalculate_preproccesed_prantl: if True, preprocessing of evey wing is
         skipped after the initial time. If False the preprocessing is repeated every iteration.
        @return: value of the equations at the point to linearize, and Jacobians w.r.t. the
         state and input vectors
        """

        if ground_wind_speed is None:
            ground_wind_speed = np.array([0, 0, 0])

        if ground_water_speed is None:
            ground_water_speed = np.array([0, 0, 0])

        kwargs = {"ground wind speed": ground_wind_speed,
                  "ground water speed": ground_water_speed,
                  "calculate ground state dot": linearize_ground_equations}

        f0, dynamics = self.calculate_state_dot(t, state_vector, input_vector, kwargs)

        if do_not_recalculate_preproccesed_prantl:
            kwargs["preprocessed main wing"] = dynamics["preprocessed main wing"]
            kwargs["preprocessed elevator"] = dynamics["preprocessed elevator"]
            kwargs["preprocessed strut"] = dynamics["preprocessed strut"]

        n = 12 if not linearize_ground_equations else 18
        a_matrix = np.zeros((n, 12))
        b_matrix = np.zeros((n, len(input_vector)))
        for pp in range(12):
            state_vector_plus = state_vector.copy()
            state_vector_plus[pp] += 1e-6

            f1, _ = self.calculate_state_dot(t, state_vector_plus, input_vector, kwargs)
            a_matrix[:, pp] = (f1 - f0) / 1e-6

        for pp in range(len(input_vector)):
            input_vector_plus = input_vector.copy()
            input_vector_plus[pp] += 1e-6
            f1, _ = self.calculate_state_dot(t, state_vector, input_vector_plus, kwargs)
            b_matrix[:, pp] = (f1 - f0) / 1e-6

        return f0, a_matrix, b_matrix
    
    def linearize_symmetric(self, state_vector, input_vector, ground_wind_speed=None,
                  ground_water_speed=None, t=0, do_not_recalculate_preproccesed_prantl=False,
                  linearize_ground_equations=False):
        """ Find the linear expansion of the governing equations at the chosen point (defined by
         the state and input vectors)
        @param state_vector: 1d, 12 elements NumPy array with the state of the boat, i.e., the
         position (x, y, z), the attitude (yaw, pitch, roll), and the linear and angular speeds
         in the boat reference frame
        @param input_vector: values of the variables that can be modified to control the boat.
        @param ground_wind_speed: 1d NumPy array with the three components of the wind speed
         w.r.t. the ground
        @param ground_water_speed: 1d NumPy array with the three components of the water speed
         w.r.t. the ground
        @param t: time, in seconds.
        @param do_not_recalculate_preproccesed_prantl: if True, preprocessing of evey wing is
         skipped after the initial time. If False the preprocessing is repeated every iteration.
        @return: value of the equations at the point to linearize, and Jacobians w.r.t. the
         state and input vectors
        """

        if ground_wind_speed is None:
            ground_wind_speed = np.array([0, 0, 0])

        if ground_water_speed is None:
            ground_water_speed = np.array([0, 0, 0])

        kwargs = {"ground wind speed": ground_wind_speed,
                  "ground water speed": ground_water_speed,
                  "calculate ground state dot": linearize_ground_equations}

        f0, dynamics = self.calculate_state_dot(t, state_vector, input_vector, kwargs)

        if do_not_recalculate_preproccesed_prantl:
            kwargs["preprocessed main wing"] = dynamics["preprocessed main wing"]
            kwargs["preprocessed elevator"] = dynamics["preprocessed elevator"]
            kwargs["preprocessed strut"] = dynamics["preprocessed strut"]

        n = 12 if not linearize_ground_equations else 18
        a_matrix = np.zeros((n, 12))
        b_matrix = np.zeros((n, len(input_vector)))
        
        delta = 1e-6
        for pp in range(12):
            state_vector_plus = state_vector.copy()
            state_vector_minus = state_vector.copy()
            state_vector_plus[pp] += delta
            state_vector_minus[pp] -= delta
            
            f1_plus, _ = self.calculate_state_dot(t, state_vector_plus, input_vector, kwargs)
            f1_minus, _ = self.calculate_state_dot(t, state_vector_minus, input_vector, kwargs)
            
            a_plus = (f1_plus - f0) / delta
            a_minus = (f1_minus - f0) / -delta
            a_mean = np.mean([a_plus, a_minus], axis=0)
            
            a_matrix[:, pp] = a_mean

        for pp in range(len(input_vector)):
            input_vector_plus = input_vector.copy()
            input_vector_minus = input_vector.copy()
            input_vector_plus[pp] += delta
            input_vector_minus[pp] -= delta
            
            f1_plus, _ = self.calculate_state_dot(t, state_vector, input_vector_plus, kwargs)
            f1_minus, _ = self.calculate_state_dot(t, state_vector, input_vector_minus, kwargs)
            
            b_plus = (f1_plus - f0) / delta
            b_minus = (f1_minus - f0) / -delta
            b_mean = np.mean([b_plus, b_minus], axis=0)
            
            b_matrix[:, pp] = b_mean

        return f0, a_matrix, b_matrix

    def stability(self, state_vector, input_vector, ground_wind_speed=None,
                  ground_water_speed=None, case_label=None,
                  do_not_recalculate_preproccesed_prantl=False):
        """ Calculates the stability of the system, providing the number of unstable and
         indifferent modes. It also plots the all the eigenvalues of the linearized system.
        @param state_vector: as in method ´self.linearize´
        @param input_vector: as in method ´self.linearize´
        @param ground_wind_speed: as in method ´self.linearize´
        @param ground_water_speed: as in method ´self.linearize´
        @param case_label: label shown in the plots
        @param do_not_recalculate_preproccesed_prantl: if True, preprocessing of evey wing is
         skipped after the initial time. If False the preprocessing is repeated every iteration.
        @return: same as method ´self.linearize´ plus a tuple with the number of unstable and
         indifferent modes
        """

        f0, a, b = self.linearize(state_vector, input_vector,
                                  ground_wind_speed=ground_wind_speed,
                                  ground_water_speed=ground_water_speed,
                                  do_not_recalculate_preproccesed_prantl=do_not_recalculate_preproccesed_prantl)

        eigenvalues, eigenvectors = np.linalg.eig(a)

        mask = np.real(eigenvalues) > 1e-10
        unstable_index_list = np.array(range(len(state_vector)))[mask]
        n_unstable = np.sum(mask)

        mask = np.logical_and(np.real(eigenvalues) > -1e-10, np.real(eigenvalues) < 1e-10)
        indifferent_index_list = np.array(range(len(state_vector)))[mask]
        n_indifferent = np.sum(mask)

        fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]])

        # Plot list of eigenvalues
        fig.add_trace(
            go.Scatter(
                visible=True,
                mode='markers',
                marker_size=16,
                marker_symbol='circle',
                marker_color='white',
                marker=dict(line=dict(color='green', width=2)),
                # line=dict(color=colors[rr], width=2),
                name="eigenvalues",
                x=np.real(eigenvalues),
                y=np.imag(eigenvalues)),
            row=1, col=1
        )
        fig.update_xaxes(title_text=r"$\Re(\lambda)$", row=1, col=1)
        fig.update_yaxes(title_text=r"$\Im(\lambda)$", row=1, col=1)

        for pp, index in enumerate(unstable_index_list):
            fig.add_trace(
                go.Bar(
                    x=self.state_labels,
                    y=np.sign(np.real(eigenvectors[:, index])) * np.abs(eigenvectors[:, index]),
                    name="{}".format(eigenvalues[index])),
                row=2, col=1
            )

        for pp, index in enumerate(indifferent_index_list):
            fig.add_trace(
                go.Bar(
                    x=self.state_labels,
                    y=np.sign(np.real(eigenvectors[:, index])) * np.abs(eigenvectors[:, index]),
                    name="{}".format(eigenvalues[index])),
                row=2, col=2
            )

        if case_label is not None:
            fig.update_layout(title=dict(text=case_label))

        fig.show()

        return f0, a, b, (n_unstable, n_indifferent)

    def _step(self, t, x, u, fixed_dofs, kwargs):
        """
        Wrapper of the method ´self.calculate_state_dot´ modified to meet the requirements of
         the solver ´scipy.solve_ivp´
        @param t: time, in seconds
        @param x: state vector
        @param u: input vector
        @param fixed_dofs: degrees of freedom that will not change along simulation
        @param kwargs: other parameters, different from the previously provided, that may be
         required by ´self.calculate_state_dot´
        @return: same as `self.calculate_state_dot`
        """
        state_dot, _ = self.calculate_state_dot(t, state_vector=x, input_vector=u, kwargs=kwargs)
        if fixed_dofs is not None:
            state_dot[fixed_dofs] = 0

        return state_dot

    def simulate(self, simulation_time, initial_state, initial_inputs, control: pyairy.Control,
                 fixed_dofs=None, ground_wind_speed=None, ground_water_speed=None,
                 total_thrust=None, differential_thrust=0.0, monitoring_parameters=None,
                 actuator_parameters=None, noise_parameters=None, wave_parameters=None,
                 results_path=None, continuation=None, do_not_recalculate_preproccesed_prantl=False,
                 use_airy=False):
        """! Conducts a temporal simulation by numerical integration of an
         Initial Value Problem described by the vehicle, environment and
         initial conditions.
        @param simulation_time: End of the integration interval
        @param initial_state: Initial conditions of the state
        @param initial_inputs: Initial conditions of the inputs
        @param control: PyAiry Control class instance
        @param fixed_dofs: List of Degrees of Freedom that will be forced to remain as provided by
         initial state. For example, this is useful to force 2D simulations by setting
         fixed_dofs = [1, 3, 5, 7, 9, 11]
        @param ground_wind_speed: 1d NumPy array with the three components of the wind speed
         w.r.t. the ground
        @param total_thrust: sum of the thrust of both propellers
        @param differential_thrust: amount of the additional thrust provided by the starboard
         propeller compared to the port one. If `T` is the total thrust of the boat,
         the starboard propeller provides `T/2 * (1 + differential_thrust)´ while the port
         propeller provides `T/2 * (1 - differential_thrust)´
        @param ground_water_speed: 1d NumPy array with the three components of the water speed
         w.r.t. the ground
        @param monitoring_parameters: a dictionary to state which variables must be shown when
         the solver is running, It contains three keys:
         * ´dt´: running time step between plots
         * ´dt dynamics´: running time step between calculation of the dynamic variables
         * ´variables´: as the input ´monitored_variables´ required by ´_monitoring_simulation´
        @param actuator_parameters: dictionary that contains the parameters required to model
         the actuator, namely, "max speed", "max acceleration", "overshoot", and "steady delay"
        @param noise_parameters: parameters to generate random noise. A better model should be
         added.
        @param wave_parameters: as in the method `_get_wave_values`
        @param results_path: path where results must be stored. If None, nothing is save to hard
         drive.
        @param continuation: if None a new simulation, from scratch, would be run. If it is a
         dictionary it must contain the keys "t0", "x0", "u0", "u1", and "stating values"
        @param do_not_recalculate_preproccesed_prantl: if True, preprocessing of evey wing is
         skipped after the initial time. If False the preprocessing is repeated every iteration.
        @param use_airy wether to use airy C control or regular python contorl mode
        @return the dictionary of results
        """

        if fixed_dofs is None:
            fixed_dofs = []

        if monitoring_parameters is not None:
            if not isinstance(monitoring_parameters, dict):
                raise ValueError("`monitoring_parameters` must be a dictionary")
            
            process_idx = monitoring_parameters.get("process index", None)
            dt_plot = monitoring_parameters.get("dt plot", None)
            dt_save = monitoring_parameters.get("dt save", np.inf)
            dt_print = monitoring_parameters.get("dt print", np.inf)
            monitored_variables = monitoring_parameters.get("variables", None)
            if (dt_plot is None) or (monitored_variables is None):
                raise ValueError("`monitoring_parameters` must have the `dt plot` and `variables` keys")

            dt_dynamics = monitoring_parameters.get("dt dynamics", np.inf)
            time_queue = monitoring_parameters.pop("time queue", None) # Queue object can't be pickled so it must be removed
            termination_event = monitoring_parameters.pop("termination event", None)
            process_index = monitoring_parameters["process index"]
        else:
            dt_plot = np.inf
            dt_save = np.inf
            dt_dynamics = np.inf
            monitored_variables = None

        def initialize_storage():
            """
            Initializes an empty database, which is a dictionary with all the
             required keys and their values are empty lists
            @return: the empty database
            """
            dynamic_variables = ["strut wave height", "strut wave vertical speed",
                                 "strut wave horizontal speed", "strut wave speed",
                                 "strut span", "total x force", "total y force", "total z force",
                                 "total x moment", "total y moment", "total z moment",
                                 "strut effective aoa lower tip",
                                 "strut effective aoa upper tip",
                                 "strut effective aoa center",
                                 "main wing effective aoa stbd tip",
                                 "main wing effective aoa port tip",
                                 "main wing effective aoa center",
                                 "elevator effective aoa stbd tip",
                                 "elevator effective aoa port tip",
                                 "elevator effective aoa center",
                                 "total thrust", "differential thrust", # My own additions
                                 "stbd propeller speed", "port propeller speed",
                                 "main wing force x", "main wing force y", "main wing force z",
                                 "main wing moment x", "main wing moment y", "main wing moment z",
                                 "elevator force x", "elevator force y", "elevator force z",
                                 "strut force x", "strut force y", "strut force z", 
                                 "stbd propeller force x", "stbd propeller force y", "stbd propeller force z",
                                 "port propeller force x", "port propeller force y", "port propeller force z",
                                 "stbd propeller moment x", "stbd propeller moment y", "stbd propeller moment z",
                                 "port propeller moment x", "port propeller moment y", "port propeller moment z"]

            database = {key: [] for key in ["t", "td"]}
            database.update({"actual " + key: [] for key in self.state_labels})
            database.update({"measured " + key: [] for key in self.state_labels})
            database.update({"commanded " + key: [] for key in self.inputs_for_this_case})
            database.update({"executed " + key: [] for key in self.inputs_for_this_case})
            database.update({"actual " + key: [] for key in self.inputs_for_this_case})
            database.update({key: [] for key in dynamic_variables})
            database.update({"airy heave": []})
            database.update({"airy pitch_sp": []})
            database.update({"commanded integrated " + key: [] for key in self.inputs_for_this_case})
            database.update({"commanded proportional " + key: [] for key in self.inputs_for_this_case})
            database.update({"setpoint " + key: [] for key in np.array(self.state_labels)[control.settings.dofs]})
            database.update({"error " + key: [] for key in np.array(self.state_labels)[control.settings.dofs]})
            return database

        def store_data(database):
            """
            Adds new data to the dictionary ´database´
            @param database: dictionary with lists of the data being monitorized
            @return: the updated database
            """

            database["t"].append(t + dt)
            for data_label, value in zip(self.state_labels, x0):
                database["actual " + data_label].append(value)
            for data_label, value in zip(self.state_labels, x):
                database["measured " + data_label].append(value)
            for data_label, value in zip(self.inputs_for_this_case, u0):
                database["commanded " + data_label].append(value)
            for data_label, value in zip(self.inputs_for_this_case, u1):
                database["executed " + data_label].append(value)
            for data_label, value in zip(self.inputs_for_this_case, u):
                database["actual " + data_label].append(value)
            try:
                database["airy heave"].append(self.estimated_heave)
                database["airy pitch_sp"].append(self.pitch_sp)
            except:
                pass
            try: # Will fail first time (before simulation loop starts)
                for data_label, value in zip(self.inputs_for_this_case, ui):
                    database["commanded integrated " + data_label].append(value)
                for data_label, value in zip(self.inputs_for_this_case, ux):
                    database["commanded proportional " + data_label].append(value)
            except:
                pass
            for idx, key in enumerate(np.array(self.state_labels)[control.settings.dofs]):
                database["error " + key].append(control.x_error[idx])
                database["setpoint " + key].append(control.x_ref[idx])

            return database

        def store_dynamics(database):
            """
            Adds the last calculated ´dynamics´ dictionary in the database of monitored
             variables.
            @param database: the dictionary with lists of all data being stored for further use.
            @return: the updated database
            """
            database["td"].append(t + dt)
            n = int(np.round(len(dynamics["strut wave"]["height"]) / 2))
            database["strut wave height"].append(dynamics["strut wave"]["height"][n])
            database["strut wave vertical speed"].append(dynamics["strut wave"]["speed"][n, 2])
            v = np.linalg.norm(dynamics["strut wave"]["speed"][n, :2])
            database["strut wave horizontal speed"].append(v)
            v = np.linalg.norm(dynamics["strut wave"]["speed"][n, :])
            database["strut wave speed"].append(v)
            database["strut span"].append(dynamics["strut span"])
            database["total x force"].append(dynamics["total force"][0])
            database["total y force"].append(dynamics["total force"][1])
            database["total z force"].append(dynamics["total force"][2])
            database["total x moment"].append(dynamics["total moment"][0])
            database["total y moment"].append(dynamics["total moment"][1])
            database["total z moment"].append(dynamics["total moment"][2])
            # My own additions
            database["total thrust"].append(dynamics["total thrust"])
            database["differential thrust"].append(dynamics["differential thrust"])
            database["stbd propeller speed"].append(dynamics["stbd propeller speed"])
            database["port propeller speed"].append(dynamics["port propeller speed"])
            database["main wing force x"].append(dynamics["main wing force"][0]) # TODO: Makes this nicer with a loop
            database["main wing force y"].append(dynamics["main wing force"][1])
            database["main wing force z"].append(dynamics["main wing force"][2])
            database["main wing moment x"].append(dynamics["main wing moment"][0])
            database["main wing moment y"].append(dynamics["main wing moment"][1])
            database["main wing moment z"].append(dynamics["main wing moment"][2])
            database["elevator force x"].append(dynamics["elevator force"][0])
            database["elevator force y"].append(dynamics["elevator force"][1])
            database["elevator force z"].append(dynamics["elevator force"][2])
            database["strut force x"].append(dynamics["strut force"][0])
            database["strut force y"].append(dynamics["strut force"][1])
            database["strut force z"].append(dynamics["strut force"][2])
            database["stbd propeller force x"].append(dynamics["stbd propeller force"][0])
            database["stbd propeller force y"].append(dynamics["stbd propeller force"][1])
            database["stbd propeller force z"].append(dynamics["stbd propeller force"][2])
            database["port propeller force x"].append(dynamics["port propeller force"][0])
            database["port propeller force y"].append(dynamics["port propeller force"][1])
            database["port propeller force z"].append(dynamics["port propeller force"][2])
            database["stbd propeller moment x"].append(dynamics["stbd propeller moment"][0])
            database["stbd propeller moment y"].append(dynamics["stbd propeller moment"][1])
            database["stbd propeller moment z"].append(dynamics["stbd propeller moment"][2])
            database["port propeller moment x"].append(dynamics["port propeller moment"][0])
            database["port propeller moment y"].append(dynamics["port propeller moment"][1])
            database["port propeller moment z"].append(dynamics["port propeller moment"][2])
            

            labels = ["strut effective aoa lower tip",
                      "strut effective aoa upper tip",
                      "strut effective aoa center"]
            indexes = [0, self.n_points["strut"]-1, int((self.n_points["strut"]-1) / 2)]
            for key, index in zip(labels, indexes):
                database[key].append(dynamics["solution strut"]["effective aoa"][index])

            labels = ["main wing effective aoa stbd tip",
                      "main wing effective aoa port tip",
                      "main wing effective aoa center"]
            indexes = [self.n_points["main wing"] - 1, 0,
                       int((self.n_points["main wing"] - 1) / 2)]
            for key, index in zip(labels, indexes):
                database[key].append(dynamics["solution main wing"]["effective aoa"][index])

            labels = ["elevator effective aoa stbd tip",
                      "elevator effective aoa port tip",
                      "elevator effective aoa center"]
            indexes = [self.n_points["elevator"] - 1, 0,
                       int((self.n_points["elevator"] - 1) / 2)]
            for key, index in zip(labels, indexes):
                database[key].append(dynamics["solution elevator"]["effective aoa"][index])

            return database

        def save_pickle_data(simulation_finished=False):
            starting_values = {key: value[-2:] for key, value in results.items()}
            continuation_params = {"t0": t, "x0": x0, "u0": u0, "u1": u1, "starting values": starting_values}
            sim_outputs = (results, x, u, continuation_params)
            
            if simulation_finished:
                res_path_finished = results_path.replace(".pkl", "-" + str(int(simulation_time)) + ".pkl")
                pickle.dump((sim_inputs, sim_outputs), open(res_path_finished, "wb"))
            else:    
                res_path_tmp = results_path.replace(".pkl", "-tmp.pkl")
                pickle.dump((sim_inputs, sim_outputs), open(res_path_tmp, "wb"))
                
                
            
        # Actuator model
        sd = control.settings.dt * actuator_parameters["steady delay"]
        use_actuator_model = sd >= 0
        if use_actuator_model:
            overshoot = actuator_parameters["overshoot"]
            a0, b0 = pyairy.match_sos(steady_delay=sd, overshoot=overshoot)[2:4]
            ad, bd = pyairy.discretize(a0, b0, dt=control.settings.dt)
            ada = np.zeros([2*initial_inputs.size, 2*initial_inputs.size])
            bda = np.zeros([2*initial_inputs.size, initial_inputs.size])
            xam = np.zeros(2 * initial_inputs.size)
            for i in range(initial_inputs.size):
                ada[2*i:2*i+2, 2*i:2*i+2] = ad
                bda[2*i:2*i+2, i] = bd
                if ~np.mod(i, 2):
                    xam[2*i] = initial_inputs[i]
            xam0 = xam.copy()
        else:
            raise ValueError("An actuator model must be provided")

        bound_v = actuator_parameters["max speed"] * np.ones(initial_inputs.size)
        bound_a = actuator_parameters["max acceleration"] * np.ones(initial_inputs.size)

        if noise_parameters is None:
            noise_parameters = {"seed": 211104, "mu_x": 1.0, "mu_u": 1.0,
                                "a_x": np.zeros(initial_state.size),
                                "a_u": np.zeros(initial_inputs.size)}

        np.random.seed(noise_parameters["seed"])
        mu_x = noise_parameters["mu_x"]
        a_x = noise_parameters["a_x"]
        mu_u = noise_parameters["mu_u"]
        a_u = noise_parameters["a_u"]

        def generate_noise(a, s):
            return a * (2 * np.random.rand(s) - 1)

        ex = (1 - mu_x) * generate_noise(a_x, initial_state.size)
        eu = (1 - mu_u) * generate_noise(a_u, initial_inputs.size)

        dt = control.settings.dt
        x = initial_state.copy()
        u = initial_inputs.copy()
        results = initialize_storage()
        if use_airy:
            imu = airy_c.InertialMeasurementUnit()
            uss_fore = airy_c.UltrasonicSensor()
            uss_aft = airy_c.UltrasonicSensor()
            acs = airy_c.FoilcartActuators()
            coc = airy_c.CompanionComputer()
            rc = airy_c.RcReceiver()
        
        dynamics = self.calculate_dynamics(0, x, u, wave_parameters=wave_parameters,
                                           ground_wind_speed=ground_wind_speed,
                                           ground_water_speed=ground_water_speed)

        if continuation is None:
            t0 = 0
            t = -dt
            t_sim = np.arange(0, simulation_time, step=dt)
            x0 = initial_state.copy()
            u0 = initial_inputs.copy()
            u1 = initial_inputs.copy()
            first_iteration = True
            results = store_data(results)
            results = store_dynamics(results)
        else:
            t0 = continuation["t0"]
            t = t0
            t_sim = np.arange(t0 + dt, t0 + simulation_time, step=dt)
            x0 = continuation["x0"]
            u0 = continuation["u0"]
            u1 = continuation["u1"]
            first_iteration = False
            results = continuation["stating values"]
            results = store_dynamics(results)

        sim_inputs = cp.deepcopy((simulation_time, initial_state, initial_inputs, control,
                        fixed_dofs, ground_wind_speed, ground_water_speed,
                        monitoring_parameters, actuator_parameters, noise_parameters,
                        wave_parameters)) # Deepcopy because initial_inputs are somehow modified otherwise     

        save_pickle_data()

        t_plot = t0
        t_print = t0
        t_save = t0
        t_dynamics = t0
        
        try:
            for t in t_sim:
                time_beginning = time.time()

                time_queue.put((process_index, t)) # Send time to the main process
                
                control.update_setpoints(t)
                    
                if do_not_recalculate_preproccesed_prantl:
                    kwargs = {"ground wind speed": ground_wind_speed,
                            "ground water speed": ground_water_speed,
                            "wave parameters": wave_parameters,
                            "total thrust": total_thrust,
                            "differential thrust": differential_thrust,
                            "preprocessed main wing": dynamics["preprocessed main wing"],
                            "preprocessed elevator": dynamics["preprocessed elevator"],
                            "preprocessed strut": dynamics["preprocessed strut"]}
                else:
                    kwargs = {"ground wind speed": ground_wind_speed,
                            "ground water speed": ground_water_speed,
                            "wave parameters": wave_parameters,
                            "total thrust": total_thrust,
                            "differential thrust": differential_thrust}

                args = (u, fixed_dofs, kwargs)
                sim = solve_ivp(self._step, t_span=(t, t + dt), y0=x, method='RK23', args=args)
                x0 = sim.y[:, -1]
                x0[fixed_dofs] = initial_state[fixed_dofs]

                ex += generate_noise(a_x, x.size)
                x = x0 + (1 - mu_x) * ex
                # Calculate control output
                if(use_airy):
                    # Accelerations are not a part of the state vector, so we need to add them manually                
                    if(t == 0):# to avoid big accelerations the first iteration
                        imu.linear.acceleration = [0, 0, 9.80665]
                        imu.angular.acceleration = [0, 0, 0]
                    else: 
                        imu.linear.acceleration =  ((imu.linear.velocity - x[6:9]) / dt) + [0, 0, 9.80665]
                        imu.angular.acceleration = (imu.angular.velocity - [x[11], x[10], x[9]]) / dt
                    imu.linear.position = x[:3]
                    imu.linear.velocity = x[6:9]
                    # Not the same convention for angles. Still NED, but different order
                    imu.angular.position = [x[5], x[4], x[3]]
                    imu.angular.velocity = [x[11], x[10], x[9]]
                    ground2boat = self.get_ground2boat_matrix(x[:6])
                    imu.linear.acceleration = ground2boat.dot(imu.linear.acceleration)
                    uss_distances = self._calculate_ultrasound_output(t, x[0:6], wave_parameters)
                    uss_fore.distance = uss_distances[0][0]
                    uss_aft.distance = uss_distances[0][1]
                    if(t == 0):
                        for i in range(500):
                            log = control.step(imu, uss_fore, uss_aft, acs, coc, rc)
                        control.reset()
                    # If you want to dinamically change the setpoint, you can do it here
                    # elif(t > 0.2 and t < 0.22):
                    #     airy_settings = control.settings
                    #     airy_settings.heave_sp = np.array([0.4, 0.9])
                    #     control.set_settings(airy_settings)

                    log = control.step(imu, uss_fore, uss_aft, acs, coc, rc)
                    self.estimated_heave = -log.heave[0]
                    self.pitch_sp = np.rad2deg(log.pitch_c)
                    u_control = control.u
                    # Airy does not have speed control. just bypass initial value
                    potential_inputs = ["main wing", "flap stbd", "flap port", "elevator", "rudder"]
                    # Only change the inputs that are  used in this case
                    used_inputs = []
                    for a in self.inputs_for_this_case:
                        try:
                            used_inputs.append(potential_inputs.index(a))
                        except:
                            pass
                    for i in range(len(used_inputs)):
                        u0[i] = u_control[used_inputs[i]]
                    
                    
                else:
                    control.step(x[control.settings.dofs], v=x[6])
                    u0 = control.u # Commanded
                    (ui, ux) = control.ui, control.ux # Extract integral and proportional input components
                    for i, label in enumerate(self.inputs_for_this_case): # Apply maximum difference filter on propeller inputs
                        if "rotation" in label:
                            max_diff = control.settings.input_limits[i][-1] # Maximum difference per second
                            u0_prev = results["commanded " + label][-1]
                            bounds = [u0_prev - max_diff * dt, u0_prev + max_diff * dt]
                            u0[i] = np.max([bounds[0], np.min([bounds[1], u0[i]])])
                            
                    u0, control.was_clipped = control.clip_inputs(u0, control.settings.input_limits) # Clip inputs to limits

                    control.u = u0 # Commanded
                    
                if use_actuator_model:
                    xam = ada @ xam0 + bda @ u0  # Actuator dynamics
                    xam0 = xam.copy()
                    u1 = xam[[2*i for i in range(initial_inputs.size)]] # Commanded -> Executed
                else:
                    u1 = u0.copy()

                if first_iteration:
                    for rr, u_p1 in enumerate(u1):
                        label = "actual " + self.inputs_for_this_case[rr]
                        u_0 = results[label][-1]
                        u_bounds = [u_0 - 0.5 * bound_a[rr] * dt ** 2,
                                    u_0 + 0.5 * bound_a[rr] * dt ** 2]
                        u1[rr] = np.max([u_bounds[0], np.min([u_bounds[1], u_p1])])
                    first_iteration = False
                else:
                    for rr, u_p1 in enumerate(u1):
                        label = "actual " + self.inputs_for_this_case[rr]
                        u_m1 = results[label][-2]
                        u_0 = results[label][-1]
                        u_bounds = [np.max([u_m1 - 2 * bound_v[rr] * dt,
                                            2 * u_0 - u_m1 - bound_a[rr] * dt ** 2]),
                                    np.min([u_m1 + 2 * bound_v[rr] * dt,
                                            2 * u_0 - u_m1 + bound_a[rr] * dt ** 2])]
                        u1[rr] = np.max([u_bounds[0], np.min([u_bounds[1], u_p1])])
                        
                eu += generate_noise(a_u, u.size)
                u = u1 + (1 - mu_u) * eu # Executed -> Actual
                u, _ = control.clip_inputs(u, control.settings.input_limits)
                            
                results = store_data(results)
                            
                if t >= t_dynamics + dt_dynamics:
                    dynamics = self.calculate_dynamics(t, x, u, wave_parameters=wave_parameters,
                                                    ground_wind_speed=ground_wind_speed,
                                                    ground_water_speed=ground_water_speed)
                    results = store_dynamics(results)
                    t_dynamics += dt_dynamics

                if (t >= t_save + dt_save and results_path is not None): # Save partial results
                    try:
                        if not termination_event.is_set():
                            save_pickle_data()
                            t_save += dt_save
                        else :
                            raise KeyboardInterrupt
                    except KeyboardInterrupt:
                        print("Keyboard interrupt detected. Saving results and exiting simulation")
                        save_pickle_data()
                        return 0
                
                if t >= t_plot + dt_plot:
                    try:
                        self._monitoring_simulation(results, monitored_variables)
                        t_plot += dt_plot
                    except Exception as e:
                        print("Error plotting during simulation: ", e)

                if t > t_print + dt_print:
                    time_end = time.time()
                    print("Time = {:.3f}. Loop: {:.3f}".format(t, time_end - time_beginning))
                    print(" x_d: {:.3f}, heave: {:.3f}".format(x[6], x[2]))
                    print(" yaw: {:.3f}, pitch: {:.3f}, roll:{:.3f}".format(x[3], np.rad2deg(x[4]), np.rad2deg(x[5])))
                    t_print += dt_print
                    
                self._check_state_boundries(x)

        except Exception as e:
            val = control.settings.sim_data["val"] 
            print(f"\nSimulation with idx {process_index} and val {val} failed with error '{e}' on line {sys.exc_info()[-1].tb_lineno}")
            return -1 
                
        if results_path is not None: save_pickle_data(simulation_finished=True)

        self._monitoring_simulation(results, monitored_variables)
        
        return 1 
    
    @staticmethod
    def _check_state_boundries(state_values):
        """! Checks if the state is out of bounds
        @param state: state vector
        @return: True if the state is out of bounds, False otherwise
        """
        dof_dict = {"x": 0, "y": 1, "z": 2, "yaw": 3, "pitch": 4, "roll": 5,
                    "u": 6, "v": 7, "w": 8, "p": 9, "q": 10, "r": 11}
        state_bounds = {"pitch": np.array([-np.deg2rad(30), np.deg2rad(30)]),
                        "roll": np.array([-np.deg2rad(60), np.deg2rad(60)])}
        
        for state, bounds in state_bounds.items():
            state_idx = dof_dict[state]
            if state_values[state_idx] < bounds[0] or state_values[state_idx] > bounds[1]:
                raise ValueError(f"{state} is out of bounds: {state_values[state_idx].round(3)} (Bounds: {bounds.round(3)})")


    @staticmethod
    def _monitoring_simulation(database, monitored_variables):
        """
        Plots some of the stored variables as a function of time.
        @param database: dictionary of the lists of values
        @param monitored_variables: nested lists of the variables to be plotted.
         The fist level contains what to be plotted in each row of the multiple plot.
         The second level indicates what to be plotted in each column of the current row.
         The third level contains the variables to be plotted.
         Example: The input [[["y"], ["u", "w"]], [["pitch"], ["q"]], [["aoa main wing", "yaw"]]]
          will plot 7 variables in 5 subplot distributed in a 3 rows by 2 columns layout. The "y"
          displacement of the CoG will be plotted in the upper-left plot, while u and w (
          velocities of the CoG along the x and z axis) are plotted together in the upper-right
          subplot. In the center row pitch and it rate of change, q, are plotted in the left and
          right figures. The lowest row only has one plot, at the left, that contains two
          variables: the angle of attack of the main wing and the yaw of the boat.
        @return: None
        """

        colors = ["red", "green", "blue", "goldenrod", "magenta", "tomato", "teal", "turquoise",
                  "orchid", "burlywood", "coral", "fuchsia", "darkcyan", "olive", "plum", "silver",
                  "salmon", "seagreen", "peru",  "aqua", "black", "greenyellow",
                  "azure", "bisque"]

        n_rows = len(monitored_variables)
        n_columns = max([len(list_of_vars) for list_of_vars in monitored_variables])
        
        subplot_titles = [" "] * (n_rows * n_columns) # Can't be empty
        words_to_remove = ["total", "actual ", "measured ", "commanded ", "executed "] # words to remove from the subplot titles

        fig = make_subplots(rows=n_rows, cols=n_columns, subplot_titles=subplot_titles)
        
        subplot_titles = [[[] for _ in range(n_columns)] for _ in range(n_rows)]

        base_state_guide = {"x": "x CoG [m]",
                            "y": "y CoG [m]",
                            "z": "z CoG [m]",
                            "yaw": "yaw [deg]",
                            "pitch": "pitch [deg]",
                            "roll": "roll [deg]",
                            "u": "u CoG [m/s]",
                            "v": "v CoG [m/s]",
                            "w": "w CoG [m/s]",
                            "p": "p [deg/s]",
                            "q": "q [deg/s]",
                            "r": "r [deg/s]"
                            }
        base_input_guide = {"main wing": "main wing [deg]",
                            "elevator": "elevator [deg]",
                            "rudder": "rudder [deg]"
                            }
        dynamics_variables = ["strut effective aoa lower tip",
                              "strut effective aoa upper tip",
                              "strut effective aoa center",
                              "main wing effective aoa stbd tip",
                              "main wing effective aoa port tip",
                              "main wing effective aoa center",
                              "elevator effective aoa stbd tip",
                              "elevator effective aoa port tip",
                              "elevator effective aoa center",
                              "strut wave height", "strut wave vertical speed",
                              "strut wave horizontal speed", "strut wave speed", "strut span",
                              "total x force", "total y force", "total z force",
                              "total x moment", "total y moment", "total z moment"]

        var_guide = {"actual " + key: "Actual " + value
                     for key, value in base_state_guide.items()}
        var_guide.update({"measured " + key: "Measured " + value
                          for key, value in base_state_guide.items()})
        var_guide.update({"commanded " + key: "Commanded " + value
                          for key, value in base_input_guide.items()})
        var_guide.update({"executed " + key: "Executed " + value
                          for key, value in base_input_guide.items()})
        var_guide.update({"actual " + key: "Actual " + value
                          for key, value in base_input_guide.items()})
        var_guide.update({v: v for v in dynamics_variables})
        var_guide.update({"airy heave": "Estimated Heave [m]"})
        var_guide.update({"airy pitch_sp": "Pitch setpoint [deg]"})
        var_guide.update({"total thrust": "Total thrust [N]"})
        turn_into_degrees = ["yaw", "pitch", "roll", "p", "q", "r"]

        counter = 0
        for pp, row_of_plots in enumerate(monitored_variables):
            for qq, cell_of_plots in enumerate(row_of_plots):
                if not isinstance(monitored_variables[pp][qq], list):
                    raise ValueError("Incorrect structure of `monitored_variables`")

                for var_alias in cell_of_plots:
                    if var_alias not in var_guide.keys():
                        raise ValueError("Incorrect variable name in `monitored_variables: `" +
                                         var_alias)

                    if var_alias not in dynamics_variables:
                        x = database["t"]
                    else:
                        x = database["td"]

                    if not any([word in turn_into_degrees for word in var_alias.split(" ")]):
                        y = database[var_alias]
                    else:
                        y = np.degrees(database[var_alias])

                    fig.add_trace(
                        go.Scatter(
                            visible=True,
                            line=dict(color=colors[counter], width=2),
                            name=var_guide[var_alias],
                            x=x,
                            y=y),
                        row=pp+1, col=qq+1)
                    
                    subplot_titles[pp][qq].append(var_alias)

                    counter += 1
                    
                subplot_titles[pp][qq] = ', '.join(subplot_titles[pp][qq]) 
                # Update the subplot titles
                if len(subplot_titles[pp][qq]) > 0: 
                    for word in words_to_remove: # Remove words from the subplot titles to make them shorter
                        subplot_titles[pp][qq] = subplot_titles[pp][qq].replace(word, "")
                    fig.layout.annotations[pp*n_rows + qq].update(text=subplot_titles[pp][qq])

        fig.update_xaxes(title_text="t [s]")

        fig.show()
        
        return fig

    def _calculate_ultrasound_output(self, t, attitude, wave_parameters, boat2ground=None,
                                     eps=1e-10):
        """
        Wrapper of ´self._calculate_position_of_interface` to input the positions of the
         ultrasounds placed in the boat. Then, this method calculates the distance measured by
         an array of ultrasound sensors (USS) placed in a boat with an arbitrary attitude
         sailing in a wavy sea
        @param t: time, in seconds
        @param attitude: concatenation of the position of the CoG (x, y, z) and the Euler angles
         of the boat (yaw, pitch, roll)
        @param wave_parameters: dictionary required by ´_get_wave_values´
        @param boat2ground: rotation matrix from boat to ground reference frames. If not
         provided, it is calculated.
        @param eps: error tolerance used by the stop criterion of the iterative solver.
        @return: a list of all the measured distances by each USS. Also a None after that.
        """

        positions = self.ultrasound_positions - self.boat["CoG [m]"]
        return self._calculate_position_of_interface(t, positions, attitude,
                                                     wave_parameters, boat2ground, eps)

    def _calculate_position_of_interface(self, t, positions, attitude, wave_parameters,
                                         boat2ground=None, eps=1e-10):
        """
        Calculates the vertical distance, in boat reference frame, from a set of points to the
         wavy sea. Used to calculate the wetted area of struts and the measures provided by
         ultrasound sensors
        @param t: time
        @param positions: list of the positions from where distances are to be calculated
        @param attitude: concatenation of the position of the CoG (x, y, z) and the Euler angles
         of the boat (yaw, pitch, roll)
        @param wave_parameters: dictionary required by ´_get_wave_values´
        @param boat2ground: rotation matrix from boat to ground reference frames. If not
         provided, it is calculated.
        @param eps: error tolerance used by the stop criterion of the iterative solver.
        @return: a list of all the distances. Also a None, because legacy.
        """

        def calculate_error(zeta, x_boat):
            """
            Given a position of the boat and a vertical distance, calculates the distance to
             the water and evaluates if the vertical distance is actually the distance to the
             water.
            @param zeta: vertical distance in the boat reference frame
            @param x_boat: position, in the boat reference frame, from wich vertical distances
             are measured.
            @return: error in the guess
            """
            
            # Raise error if any warning is raised
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    x_beam_boat = x_boat + zeta * np.array([0, 0, 1])
                    x_ground = np.array([boat2ground.dot(x_beam_boat)]) + attitude[:3]
                    z_w, _, _ = self._get_wave_values(t, x_ground, wave_parameters,
                                                    only_provide_h=True)
                except Warning as e:
                    raise ValueError("Error in calculate_error: ", e)
            
            return x_ground[0, 2] + z_w[0]

        if boat2ground is None:
            boat2ground = self.get_boat2ground_matrix(attitude)

        measures = []
        height_from_cog = []
        for xs in positions:
            zeta_w = 0
            xs_boat = xs.copy()
            xw_boat = xs_boat
            while True:
                error = calculate_error(zeta_w, xw_boat)
                error_plus = calculate_error(zeta_w + 1e-6, xw_boat)
                d_error_d_zeta = (error_plus - error) / 1e-6
                d_zeta = -error / d_error_d_zeta
                zeta_w += d_zeta

                if np.abs(error) < eps:
                    measures.append(zeta_w - xs_boat[2])
                    height_from_cog.append(zeta_w)
                    break

        return measures, None

    def calculate_dynamics(self, t, state_vector, input_vector, wave_parameters=None,
                           ground_wind_speed=None, ground_water_speed=None, print_dynamics=False):
        """
        Wrapper to calculate the forces and moments over the boat as done by the method
         ´self.calculate_forces_and_moments´
        @param t: time
        @param state_vector: 1d, 12 elements, NumPy array with the state of the boat, i.e.,
        the position
         (x, y, z), the attitude (yaw, pitch, roll), and the linear and angular speeds in the
         boat reference frame.
        @param input_vector: values of the variables that can be modified to control the boat.
        @param wave_parameters: as in the method `_get_wave_values`
        @param ground_wind_speed: 1d NumPy array with the three components of the wind speed
         w.r.t. the ground
        @param ground_water_speed: 1d NumPy array with the three components of the water speed
         w.r.t. the ground
        @param print_dynamics: if True, the values of dynamics are shown by screen
        @return: dictionary ´dynamics´ as returned by ´self.calculate_forces_and_moments´
        """

        attitude, attitude_dot = self.split_state(state_vector)

        wing_angles = {"main wing": 0, "elevator": 0}
        flap_angles = {"main wing": {"stbd": 0, "port": 0}, "strut": {"upper": 0, "lower": 0}}
        thrust_params = {}
        input_labels = self.inputs_for_this_case + list(self.fixed_inputs.keys())
        input_vector = np.concatenate((input_vector, list(self.fixed_inputs.values())))
        for label, value in zip(input_labels, input_vector):
            if label in ["main wing", "elevator"]:
                wing_angles[label] = value
            elif label == "flap stbd":
                flap_angles["main wing"]["stbd"] = value
            elif label == "flap port":
                flap_angles["main wing"]["port"] = value
            elif label == "flap symmetric":
                flap_angles["main wing"]["stbd"] = value
                flap_angles["main wing"]["port"] = -value
            elif label == "flap antisymmetric":
                flap_angles["main wing"]["stbd"] = value
                flap_angles["main wing"]["port"] = value
            elif label == "half rudder":
                flap_angles["strut"]["lower"] = value
            elif label == "rudder" or label == "whole rudder":
                flap_angles["strut"]["upper"] = value
                flap_angles["strut"]["lower"] = value
            elif label == "total thrust":
                thrust_params["type"] = "constant thrust"
                thrust_params["thrust [N]"] =  value
            elif label == "differential thrust":
                thrust_params["differential thrust"] = value
            elif label == "stbd rotation speed":
                thrust_params["type"] = "propellers"
                thrust_params["stbd propeller"] = value
            elif label == "port rotation speed":
                thrust_params["type"] = "propellers"
                thrust_params["port propeller"] = value
            elif label == "propellers rotation speed":
                thrust_params["type"] = "propellers"
                thrust_params["stbd propeller"] = value
                thrust_params["port propeller"] = value
            elif label == "average rotation speed":
                thrust_params["type"] = "propellers"
                if "differential rotation speed" in input_labels:
                    differential_rotation_speed = input_vector[input_labels.index("differential rotation speed")]
                    thrust_params["stbd propeller"] = value * (1 + differential_rotation_speed)
                    thrust_params["port propeller"] = value * (1 - differential_rotation_speed)
                else:
                    thrust_params["stbd propeller"] = value
                    thrust_params["port propeller"] = value

        if thrust_params.get("type", None) is None:
            thrust_params["type"] = "constant speed"

        if ground_wind_speed is None:
            ground_wind_speed = np.array([0, 0, 0])
        if ground_water_speed is None:
            ground_water_speed = np.array([0, 0, 0])

        dynamics = self.calculate_forces_and_moments(t=t,
                                                     attitude=attitude,
                                                     attitude_dot=attitude_dot,
                                                     wing_angles=wing_angles,
                                                     flap_angles=flap_angles,
                                                     wave_parameters=wave_parameters,
                                                     ground_wind_speed=ground_wind_speed,
                                                     ground_water_speed=ground_water_speed,
                                                     thrust_params=thrust_params)

        forces = ["weight", "main wing force", "main wing moment", "elevator force",
                  "elevator moment", "strut force", "strut moment", "hull force", "hull moment",
                  "fuselage force", "fuselage moment", "stbd propeller force",
                  "stbd propeller moment", "stbd motor power", "port propeller force",
                  "port propeller moment", "port motor power", "total force", "total moment"]
        if print_dynamics:
            for key in forces:
                print(key, dynamics[key])

        return dynamics

    @staticmethod
    def get_foilcart_object(new_fc_object=False, fc_obj_path='data/foilcart.pkl', args=(15, 15, 101325)): 
        """
        If no 'foilcart.pkl' file is found or if new_fc_object is true, a new instance of the
        Foilcart class is created and saved as 'foilcart.pkl
        @param new_fc_object: if True, a new instance of the Foilcart class is created and saved
        @param fc_obj_path: path to the foilcart object
        @param args: arguments to be passed to the Foilcart class: (water temp, air temp, atm pressure)
         """
        if not new_fc_object:
            try:
                with open(fc_obj_path, 'rb') as f:
                    foilcart = pickle.load(f)
                # print("Foilcart object loaded from " + fc_obj_path)
                return foilcart
            except Exception as e:
                print("Error loading Foilcart object from " + fc_obj_path + ": " + str(e))
            
        print("Initializing Foilcart class...")
        foilcart = Foilcart(*args)
        print("Foilcart class has been initialized")
        with open(fc_obj_path, 'wb') as f:
            pickle.dump(foilcart, f)
        print("Foilcart object saved at " + fc_obj_path)

        return foilcart
        
    def calc_gain_matrix(self, state_vector, input_vector, dofs, dofs_i_local, q, r, n=None, dt=0.01, 
                         only_k=False, linearization_mode="normal"):
        if linearization_mode == 'symmetric':
            f0, a, b = self.linearize_symmetric(state_vector=state_vector, input_vector=input_vector)
        else:
            f0, a, b = self.linearize(state_vector=state_vector, input_vector=input_vector)
        
        ad, bd = pyairy.discretize(a[dofs, :][:, dofs], b[dofs, :], dt=dt)

        k = pyairy.dlqri(ad, bd, q=q, r=r, n=n, dofs_i_local=dofs_i_local, dt=dt, bryson=True)[0]

        if only_k:
            return k
        else:
            return k, (f0, a, b, ad, bd)
