# For recording the stats of the actions that are chosen.
import numpy as np

class action_taken:
    def __init__(self, prev_action, static_stats, \
    specific_stats, q_hat, step_number, frac_change_in_opt_gap):
        self.prev_action = prev_action
        self.static_stats = static_stats
        self.specific_stats = specific_stats
        self.q_hat = q_hat
        self.step_number = step_number
        self.frac_change_in_opt_gap = frac_change_in_opt_gap
        
      
      
      
      
      
      
      
      
                

