"""
https://realpython.com/intro-to-python-threading/#what-is-a-thread

What Is a Thread?
A thread is a separate flow of execution. This means that your program will have two things 
happening at once. But for most Python 3 implementations the different threads do not 
actually execute at the same time: they merely appear to.

It’s tempting to think of threading as having two (or more) different processors 
running on your program, each one doing an independent task at the same time. 
That’s almost right. The threads may be running on different processors, 
but they will only be running one at a time.

Getting multiple tasks running simultaneously requires a non-standard implementation of Python, 
writing some of your code in a different language, 
or using multiprocessing which comes with some extra overhead.

Because of the way CPython implementation of Python works, threading may not speed up 
all tasks. This is due to interactions with the GIL that essentially limit one Python thread 
to run at a time.

Tasks that spend much of their time waiting for external events are generally good candidates 
for threading. Problems that require heavy CPU computation and spend little time waiting for 
external events might not run faster at all.

This is true for code written in Python and running on the standard CPython implementation. 
If your threads are written in C they have the ability to release the GIL and run concurrently. 
If you are running on a different Python implementation, check with the documentation too see 
how it handles threads.

If you are running a standard Python implementation, writing in only Python, 
and have a CPU-bound problem, you should check out the multiprocessing module instead.

"""
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

# When you create a Thread, you pass it a function and a list containing the arguments 
# to that function. In this case, you’re telling the Thread to run thread_function() 
# and to pass it 1 as an argument.    
    logging.info("Main    : before creating thread")
    x = threading.Thread( target=thread_function, args=(7,) ) # , daemon=True
    logging.info("Main    : before running thread")
    
    x.start()
    logging.info("Main    : wait for the thread to finish")
    
    x.join()
    # To tell one thread to wait for another thread to finish, you call .join(). 
    # If you uncomment that line, the main thread will pause and wait for the thread x 
    # to complete running.

    logging.info("Main    : all done")
# You’ll notice that the Thread finished after the Main section of your code did.
# This pause is Python waiting for the non-daemonic thread to complete. 
# When your Python program ends, part of the shutdown process is 
# to clean up the threading routine.

# Daemon Threads
"""
In computer science, a daemon is a process that runs in the background.

Python threading has a more specific meaning for daemon. A daemon thread will shut down 
immediately when the program exits. One way to think about these definitions is 
to consider the daemon thread a thread that runs in the background without worrying about 
shutting it down.

If a program is running Threads that are not daemons, then the program will wait for 
those threads to complete before it terminates. Threads that are daemons, however, 
are just killed wherever they are when the program is exiting.
"""
