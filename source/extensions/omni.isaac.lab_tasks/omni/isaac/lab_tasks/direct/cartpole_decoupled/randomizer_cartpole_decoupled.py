import os.path
from pathlib import Path
import yaml
import torch

class CartPoleDecoupledRandomizer:
    def __init__(self):

        cwd = Path(__file__).parent
        dr_config_path = os.path.abspath(cwd / "agents/domain_randomization_cfg.yaml")

        with open(dr_config_path, encoding="utf-8") as f:
            dr_cfg = yaml.full_load(f)

        self.dr_cfg = dr_cfg
        if self.dr_cfg is not None:
            if 'randomize' in self.dr_cfg['custom_domain_randomization']:
                if self.dr_cfg['custom_domain_randomization']['randomize']:
                    self.custom_randomize = True
                    self.randomization_params = self.dr_cfg['custom_domain_randomization']['randomization_parameters']
                else:
                    print("Custom domain set to 'False'.")
            else:
                print("Please specify if you would like to randomize.")
        else:
            print("Custom domain randomization specification is empty.")

        return

    def randomizer(self, task):

        custom_attributes = ['cart_mass', 'pole_len', 'pole_mass', 'pole_friction', 'moment_of_inertia', 'gravity']

        if self.custom_randomize:
            if self.randomization_params is not None:
                for attr in self.randomization_params:
                    if attr not in custom_attributes:
                        print(f'{attr} is not a valid attribute for the custom domain randomization.')
                if 'cart_mass' in self.randomization_params:
                    if 'bounds' in self.randomization_params['cart_mass']:
                        low = self.randomization_params['cart_mass']['bounds'][0]
                        high = self.randomization_params['cart_mass']['bounds'][1]
                        task.cart_mass *= ((low - high) * torch.rand(size=(task.num_envs,),
                                                   device=task.device) + high)
                    else:
                        print('Please specify bounds of randomization for cart mass.')
                if 'pole_len' in self.randomization_params:
                    if 'bounds' in self.randomization_params['pole_len']:
                        low = self.randomization_params['pole_len']['bounds'][0]
                        high = self.randomization_params['pole_len']['bounds'][1]
                        task.pole_len *= ((low - high) * torch.rand(size=(task.num_envs,),
                                                   device=task.device) + high)
                    else:
                        print('Please specify bounds of randomization for cart mass.')
                if 'pole_mass' in self.randomization_params:
                    if 'bounds' in self.randomization_params['pole_mass']:
                        low = self.randomization_params['pole_mass']['bounds'][0]
                        high = self.randomization_params['pole_mass']['bounds'][1]
                        task.pole_mass *= ((low - high) * torch.rand(size=(task.num_envs,),
                                                   device=task.device) + high)
                    else:
                        print('Please specify bounds of randomization for cart mass.')
                if 'pole_friction' in self.randomization_params:
                    if 'bounds' in self.randomization_params['pole_friction']:
                        low = self.randomization_params['pole_friction']['bounds'][0]
                        high = self.randomization_params['pole_friction']['bounds'][1]
                        task.pole_friction *= ((low - high) * torch.rand(size=(task.num_envs,),
                                                   device=task.device) + high)
                    else:
                        print('Please specify bounds of randomization for cart mass.')
                if 'moment_of_inertia' in self.randomization_params:
                    if 'bounds' in self.randomization_params['moment_of_inertia']:
                        low = self.randomization_params['moment_of_inertia']['bounds'][0]
                        high = self.randomization_params['moment_of_inertia']['bounds'][1]
                        task.moment_of_inertia *= ((low - high) * torch.rand(size=(task.num_envs,),
                                                   device=task.device) + high)
                    else:
                        print('Please specify bounds of randomization for cart mass.')
                if 'gravity' in self.randomization_params:
                    if 'bounds' in self.randomization_params['gravity']:
                        low = self.randomization_params['gravity']['bounds'][0]
                        high = self.randomization_params['gravity']['bounds'][1]
                        task.gravity *= ((low - high) * torch.rand(size=(task.num_envs,),
                                                   device=task.device) + high)
                    else:
                        print('Please specify bounds of randomization for cart mass.')
            else:
                print('No attributes are selected to be randomized.')

        return



