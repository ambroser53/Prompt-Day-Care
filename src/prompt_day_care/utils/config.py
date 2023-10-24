import json
import os
from typing import Dict, Any

class Config:
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.__dict__.update(kwargs)

    @classmethod
    def load_config(cls, path: str) -> 'Config':
        if not os.path.exists(path):
            raise FileNotFoundError(f'No config file found at {path}')
        else:
            with open(path, 'r') as f:
                config = json.load(f)
            
        return cls(**config)
    
    def pop(self, key: str) -> Any:
        return self.__dict__.pop(key)
    
    def add(self, key: str, value: Any) -> None:
        self.__dict__[key] = value
    
    def save_config(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)