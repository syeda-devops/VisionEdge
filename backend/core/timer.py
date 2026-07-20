import time 
 
 
class Timer: 
 
    def __init__(self): 
        self.start_time = None 
 
    def start(self): 
        self.start_time = time.perf_counter() 
 
    def stop(self): 
 
        if self.start_time is None: 
            raise RuntimeError( 
                "Timer has not been started." 
            ) 
 
        end_time = time.perf_counter() 
 
        elapsed = end_time - self.start_time 
 
        self.start_time = None 
 
        return elapsed