# For recording the stats of the actions that are chosen.
import numpy as np
class action_taken:
        def __init__(self, alpha, prev_action,  branch_or_search, static_stats, specific_stats, q_hat, step_number):
                self.alpha = alpha
                self.prev_action = prev_action
                self.branch_or_search = branch_or_search
                self.static_stats = static_stats
                self.specific_stats = specific_stats
                self.q_hat = q_hat
                self.step_number = step_number
                self.cost_of_action = 2*(branch_or_search == 'branch') + self.alpha*(branch_or_search == 'search')
                if prev_action is None:
                        self.cost_so_far = self.cost_of_action
                else:
                        self.cost_so_far = prev_action.cost_so_far + self.cost_of_action

        def get_record(self, total_cost):
                q_array = np.array([self.q_hat, total_cost - self.cost_so_far], dtype=float)
                record = np.concatenate([self.static_stats, self.specific_stats, q_array])
                return(record)

