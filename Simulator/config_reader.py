import json

class CommonConfig:
    """
    Summary: Configration data container for the overall program
    """

    def __init__(self, common_config):

        self.time_step = common_config["time_step"]
        self.start_hat = common_config["start_hat"]
        self.goal = common_config["goal"]
        self.init_cov_val = common_config["init_cov_val"]
        self.iteration_cap = common_config["iteration_cap"]

class StraightLinePlannerConfig:

    """
    Summary: Configration data container for the straight line planner
    """

    def __init__(self, straight_line_planner_config):
        self.velocity = straight_line_planner_config['velocity']

class EnvironmentConfig:

    """
    Summary: Configration data container for the environment
    """

    def __init__(self, environment_config):

        self.env_limits = environment_config["env_limits"]
        self.viz_resolution = environment_config["viz_resolution"]
        self.light_coord_x = environment_config["light_coord_x"]

class CostFunctionConfig:

    """
    Summary: Configration data container for the cost function
    """

    def __init__(self, environment_config):

        self.stage_cost_multiplier = environment_config["stage_cost_multiplier"]
        self.terminal_cost_multiplier = environment_config["terminal_cost_multiplier"]

class ConfigParser:

    """
    Summary: Parses the input json configuration file into data containers to be injected into further objects
    """

    environment_config = None
    common_config = None
    straight_line_planner_config = None
    cost_function_config = None

    def __init__(self, config_file_path):
        self._read_config_file(config_file_path)

    def _read_config_file(self, config_file_path):
        with open(config_file_path) as config_file:
            json_data = json.load(config_file)
        
        self.read_common_config(json_data)
        self.read_environment_config(json_data)
        self.read_straight_line_planner_config(json_data)
        self.read_cost_function_config(json_data)

    def read_common_config(self, json_data):
        self.common_config = CommonConfig(json_data['common_config'])

    def read_environment_config(self, json_data):
        self.environment_config = EnvironmentConfig(json_data['environment_config'])

    def read_straight_line_planner_config(self, json_data):
        self.straight_line_planner_config = StraightLinePlannerConfig(json_data['straight_line_planner_config'])

    def read_cost_function_config(self, json_data):
        self.cost_function_config = CostFunctionConfig(json_data['cost_function_config'])
