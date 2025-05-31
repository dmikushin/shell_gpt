from abc import ABC, abstractmethod
from pathlib import Path
import toml

class BaseModelConfigManager(ABC):
    """
    Abstract base class for managing model configuration.
    Subclasses must implement list_models.
    """

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def get_default_model(self):
        """
        Retrieves the default model configuration.

        Returns:
            dict or None: The default model from config file or the first available model
            or None if no valid model is                        model_tuple = (provider, name, model_type)
 found.
        """
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = toml.load(f)
            model = config.get("default_model")
            if model:
                provider = model.get("provider", "").lower()
                name = model.get("name", "").lower()
                model_type = model.get("type", "").lower()
                if provider and name and model_type:
                    return {"provider": provider, "name": name, "type": model_type}

        # Default model not set (or set incorrectly),
        # try to find available models and set the first one
        try:
            models = self.list_models(all_models=True)
            if models:
                # Find the first model with valid provider, name, and model_type
                for model in models:
                    provider = model.get("provider", "").lower()
                    name = model.get("name", "").lower()
                    model_type = model.get("type", "").lower()
                    if provider and name and model_type:
                        model = {"provider": provider, "name": name, "type": model_type}
                        self.set_default_model(model)
                        return model
        except Exception:
            pass

        return None

    def set_default_model(self, model):
        """
        Set the default model in the TOML config file.
        model: (provider, name, model_type)
        """
        # Load existing config if present
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = toml.load(f)
        else:
            config = {}

        # Unpack model tuple
        provider, name, model_type = model

        # Set default model info
        config["default_model"] = {
            "provider": provider,
            "name": name,
            "type": model_type
        }

        # Write back to TOML file
        with open(self.config_path, "w") as f:
            toml.dump(config, f)

    @abstractmethod
    def list_models(self, all_models=True):
        """
        Abstract method to list models.
        Args:
            all_models (bool): If False, fetch local models only.
        Returns:
            dict: Dictionary containing available models.
        """
        pass

    @staticmethod
    def _find_matching_models(model_name, models):
        """
        Find models matching the given model_name.
        Returns a tuple (exact_match, name_matches, ollama_matches).
        """
        if not isinstance(models, list) or not models:
            return None, [], []

        matches = []
        exact_match = None

        for model in models:
            provider = model.get("provider", "")
            name = model.get("name", "")
            model_type = model.get("type", "")
            if not provider or not name or not model_type:
                continue
            full_name = f"{provider}/{name}"
            if model_name.lower() == full_name.lower():
                exact_match = (provider, name, model_type)
                break
            matches.append((provider, name, model_type))

        name_matches = [m for m in matches if m[1].lower() == model_name.lower()]
        ollama_matches = [m for m in name_matches if m[0] == "ollama"]

        return exact_match, name_matches, ollama_matches

    def find_matching_models(self, model_name):
        return self._find_matching_models(model_name, self.list_models(all_models=True))

    def validate_model(self, model):
        """Validate the model format."""
        if model is None:
            return True  # None is valid, means use default
        
        if not isinstance(model, dict):
            return False
        
        required_fields = ['provider', 'name', 'type']
        for field in required_fields:
            if field not in model:
                return False
            if not isinstance(model[field], str) or not model[field].strip():
                return False
        
        return True

    def model_exists(self, model) -> bool:
        """
        Check if the given model exists in the available models.

        Args:
            model (dict): Model dictionary with 'provider', 'name', and 'type'.

        Returns:
            bool: True if the model is valid and exists, False otherwise.
        """
        if not self.validate_model(model):
            return False
        models = self.list_models(all_models=True)
        for m in models:
            if (
                m.get("provider", "") == model.get("provider")
                and m.get("name", "") == model.get("name")
                and m.get("type", "") == model.get("type")
            ):
                return True
        return False
