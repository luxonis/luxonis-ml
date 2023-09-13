import yaml
import json
import jsonschema

class RHConfig:

    def __init__(self, config_path, schema_path, config=None):
        self.config_path = config_path
        self.schema_path = schema_path

        if config is None:
            # get config and schema from files
            self.config, self.schema = self.get_config_and_schema(config_path, schema_path)
            self.rh_config_validate(self.config, self.schema)
            self.config = self.set_defaults_from_schema(self.config, self.schema)
        else:
            # get config from input
            self.config = config
            self.schema = self.read_schema_json(schema_path)
            self.rh_config_validate(self.config, self.schema)
            self.config = self.set_defaults_from_schema(self.config, self.schema)
    
    def get_config(self):
        return self.config
    
    def get_schema(self):
        return self.schema
    
    def get_config_path(self):
        return self.config_path
    
    def get_schema_path(self):
        return self.schema_path
    
    def pretty_print_config(self):
        print(json.dumps(self.config, indent=4))
    

    def read_schema_json(self, schema_path):
        with open(schema_path, 'r') as file:
            schema = json.load(file)
        return schema

    def read_config_yaml(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_config_and_schema(self, config_path, schema_path):
        # Read the schema
        schema = self.read_schema_json(schema_path)

        # Read the config
        config = self.read_config_yaml(config_path)

        return config, schema

    def validate_config(self, config, schema):
        try:
            jsonschema.validate(config, schema)
        except jsonschema.exceptions.ValidationError as err:
            print(err)
            return False
        return True

    def rh_config_validate(self, config, schema):
        # Convert the dates to strings
        if "from_date" in config and config["from_date"]:
            config["from_date"] = config["from_date"].strftime("%Y-%m-%d")
        if "to_date" in config and config["to_date"]:
            config["to_date"] = config["to_date"].strftime("%Y-%m-%d")

        # Validate the config
        if not self.validate_config(config, schema):
            print("Invalid config!")
            exit(1)
        else:
            print("Valid config!")

    def extract_defaults_from_schema(self, schema):
        """Recursively extract default values from the schema."""
        defaults = {}
        for key, value in schema.get("properties", {}).items():
            if "default" in value:
                defaults[key] = value["default"]
            if value.get("type") == "object":
                defaults[key] = self.extract_defaults_from_schema(value)
        return defaults

    def set_defaults_from_schema(self, config, schema):
        """Set default values for missing keys in the configuration using the schema."""
        defaults = self.extract_defaults_from_schema(schema)
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
            elif isinstance(default_value, dict):  # If the default value is a nested dictionary
                for sub_key, sub_default_value in default_value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_default_value
        return config
    
    # make object serializable
    def to_dict(self):
        """Serializes the RHConfig object into a dictionary."""
        return {
            'config': self.config,
            'schema': self.schema,
            'config_path': self.config_path,
            'schema_path': self.schema_path,
        }
    
    @classmethod
    def from_dict(cls, serialized_data):
        """Deserializes the dictionary into a new RHConfig object."""
        print("SERIALIZED DATA:  ",serialized_data)
        rh_config = cls(serialized_data['config_path'], serialized_data['schema_path'], serialized_data['config'])
        rh_config.schema = serialized_data['schema']
        return rh_config