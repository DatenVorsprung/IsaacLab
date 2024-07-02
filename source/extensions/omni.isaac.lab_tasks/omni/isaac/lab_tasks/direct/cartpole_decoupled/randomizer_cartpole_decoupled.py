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
        self.attribute_randomize = False
        self.action_randomize = False
        self.observation_randomize = False
        if self.dr_cfg is not None:
            if 'randomize' in self.dr_cfg['custom_domain_randomization']:
                if self.dr_cfg['custom_domain_randomization']['randomize']:
                    if self.dr_cfg['custom_domain_randomization']['attributes']:
                        self.attribute_randomize = True
                        self.attribute_params = self.dr_cfg['custom_domain_randomization']['attributes']
                    if self.dr_cfg['custom_domain_randomization']['action']:
                        self.action_randomize = True
                        self.action_params = self.dr_cfg['custom_domain_randomization']['action']
                    if self.dr_cfg['custom_domain_randomization']['observation']:
                        self.observation_randomize = True
                        self.observation_params = self.dr_cfg['custom_domain_randomization']['observation']
                else:
                    print("Custom randomization is set to False.")
            else:
                print("Please specify if you would like to randomize.")
        else:
            print("Custom randomization specifications are empty.")

        return

    def attribute_randomizer(self, task):

        dict_attributes = {'cart_mass': task.cart_mass, 'pole_len': task.pole_len, 'pole_mass': task.pole_mass,
                           'pole_friction': task.pole_friction, 'moment_of_inertia': task.moment_of_inertia,
                           'gravity': task.gravity}

        custom_distributions = ['uniform', 'gaussian']
        custom_operations = ['additive', 'scaling']

        if self.attribute_randomize:
            if self.attribute_params is not None:
                for attr in self.attribute_params.keys():
                    if attr not in dict_attributes.keys():
                        print(f'{attr} is not a valid attribute for the custom domain randomization.')
                for attr in dict_attributes.keys():
                    if attr in self.attribute_params.keys():
                        if not set(('operation', 'distribution', 'distribution_params')).issubset(
                                self.attribute_params[attr].keys()):
                            print('Please provide all ingredients for randomizing the action: operation, distribution, distribution_params.')
                        else:
                            if not self.action_params['distribution'] in custom_distributions:
                                print('Please select one of the two distributions: uniform, gaussian.')
                            else:
                                if self.attribute_params[attr]['distribution'] == 'uniform':
                                    low = self.attribute_params[attr]['distribution_params'][0]
                                    high = self.attribute_params[attr]['distribution_params'][1]
                                    attribute_noise = ((low - high) * torch.rand(size=dict_attributes[attr].shape, device=task.device) + high)
                                    if not self.action_params['operation'] in custom_operations:
                                        print('Please select one of the two operations: additive, scaling.')
                                    else:
                                        if self.attribute_params[attr]['operation'] == 'additive':
                                            dict_attributes[attr] += attribute_noise
                                        if self.attribute_params[attr]['operation'] == 'scaling':
                                            dict_attributes[attr] *= attribute_noise
                                if self.attribute_params[attr]['distribution'] == 'gaussian':
                                    mean = self.attribute_params[attr]['distribution_params'][0]
                                    sd = self.attribute_params[attr]['distribution_params'][1]
                                    action_noise = torch.normal(mean=(mean * torch.ones(size=dict_attributes[attr].shape, device=task.device)), std=(sd * torch.ones(size=dict_attributes[attr].shape, device=task.device)))
                                    if not self.attribute_params[attr]['operation'] in custom_operations:
                                        print('Please select one of the two operations: additive, scaling.')
                                    else:
                                        if self.attribute_params[attr]['operation'] == 'additive':
                                            dict_attributes[attr] += action_noise
                                        if self.attribute_params[attr]['operation'] == 'scaling':
                                            dict_attributes[attr] *= action_noise
            else:
                print('No attributes are selected to be randomized.')

        return

    def action_randomizer(self, task, action):

        custom_distributions = ['uniform', 'gaussian']
        custom_operations = ['additive', 'scaling']

        if self.action_randomize:
            if not set(('operation', 'distribution', 'distribution_params')).issubset(self.action_params.keys()):
                print('Please provide all ingredients for randomizing the action: operation, distribution, distribution_params.')
            else:
                if not self.action_params['distribution'] in custom_distributions:
                    print('Please select one of the two distributions: uniform, gaussian.')
                else:
                    if self.action_params['distribution'] == 'uniform':
                        low = self.action_params['distribution_params'][0]
                        high = self.action_params['distribution_params'][1]
                        action_noise = ((low - high) * torch.rand(size=action.shape, device=task.device) + high)
                        if not self.action_params['operation'] in custom_operations:
                            print('Please select one of the two operations: additive, scaling.')
                        else:
                            if self.action_params['operation'] == 'additive':
                                action += action_noise
                            if self.action_params['operation'] == 'scaling':
                                action *= action_noise

                    if self.action_params['distribution'] == 'gaussian':
                        mean = self.action_params['distribution_params'][0]
                        sd = self.action_params['distribution_params'][1]
                        action_noise = torch.normal(mean=(mean * torch.ones(size=action.shape, device=task.device)), std=(sd * torch.ones(size=action.shape, device=task.device)))
                        if not self.action_params['operation'] in custom_operations:
                            print('Please select one of the two operations: additive, scaling.')
                        else:
                            if self.action_params['operation'] == 'additive':
                                action += action_noise
                            if self.action_params['operation'] == 'scaling':
                                action *= action_noise
        return action

    def observation_randomizer(self, task, observation):

        custom_distributions = ['uniform', 'gaussian']
        custom_operations = ['additive', 'scaling']

        if self.observation_randomize:
            if not set(('operation', 'distribution', 'distribution_params')).issubset(self.observation_params.keys()):
                print('Please provide all ingredients for randomizing the observation: operation, distribution, distribution_params.')
            else:
                if not self.observation_params['distribution'] in custom_distributions:
                    print('Please select one of the two distributions: uniform, gaussian.')
                else:
                    if self.observation_params['distribution'] == 'uniform':
                        low = self.observation_params['distribution_params'][0]
                        high = self.observation_params['distribution_params'][1]
                        observation_noise = ((low - high) * torch.rand(size=observation.shape, device=task.device) + high)
                        if not self.observation_params['operation'] in custom_operations:
                            print('Please select one of the two operations: additive, scaling.')
                        else:
                            if self.observation_params['operation'] == 'additive':
                                observation += observation_noise
                            if self.observation_params['operation'] == 'scaling':
                                observation *= observation_noise

                    if self.observation_params['distribution'] == 'gaussian':
                        mean = self.observation_params['distribution_params'][0]
                        sd = self.observation_params['distribution_params'][1]
                        observation_noise = torch.normal(mean=(mean * torch.ones(size=observation.shape, device=task.device)), std=(sd * torch.ones(size=observation.shape, device=task.device)))
                        if not self.observation_params['operation'] in custom_operations:
                            print('Please select one of the two operations: additive, scaling.')
                        else:
                            if self.observation_params['operation'] == 'additive':
                                observation += observation_noise
                            if self.observation_params['operation'] == 'scaling':
                                observation *= observation_noise
        return observation


