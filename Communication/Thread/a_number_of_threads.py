import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    # The program keeps a list of Thread objects so that it can then wait for them later 
    # using .join().
    threads = list()

    for index in range(4):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=thread_function, args=(index,))
        threads.append(x)
        x.start()

    # The order in which threads are run is determined by the operating system 
    # and can be quite hard to predict. It may (and likely will) vary from run to run, 
    # so you need to be aware of that when you design algorithms that use threading.
    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)
