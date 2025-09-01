import os
from typing import Dict, Any

class EnvironmentConfig:
    """
    Environment configuration for different API endpoints and settings.
    Supports AU1 (production), TEST (via VPN), and custom environments.
    """
    
    ENVIRONMENTS = {
        "AU1": {
            "name": "AU1 Production",
            "metadata_url": "https://au1.aconex.com/mobile/rest/projects/26851/metadata/documents",
            "search_url": "https://au1.aconex.com/mobile/rest/projects/26851/documents/search/filter?content_search=false&sort_direction=DESC&page_number=1&sort=registered",
            "project_id": "26851",
            "description": "AU1 production environment - direct access"
        },
        "TEST": {
            "name": "Test Environment", 
            "metadata_url": "https://ocipl12.aconex.oraclecloud.com/mobile/rest/projects/1879053290/metadata/documents",
            "search_url": "https://ocipl12.aconex.oraclecloud.com/mobile/rest/projects/1879053290/documents/search/filter?content_search=false&sort_direction=DESC&page_number=1&sort=registered",
            "project_id": "1879053290",
            "description": "Test environment - requires VPN connection"
        }
    }
    
    def __init__(self):
        # Get environment from env variable or default to AU1
        self.current_env = os.getenv("ACONEX_ENV", "TEST").upper()
        
        # Validate environment
        if self.current_env not in self.ENVIRONMENTS:
            available = ", ".join(self.ENVIRONMENTS.keys())
            raise ValueError(f"Invalid environment '{self.current_env}'. Available: {available}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current environment configuration."""
        return self.ENVIRONMENTS[self.current_env].copy()
    
    def get_metadata_url(self) -> str:
        """Get the metadata API URL for current environment."""
        return self.ENVIRONMENTS[self.current_env]["metadata_url"]
    
    def get_search_url(self) -> str:
        """Get the search API URL for current environment."""
        return self.ENVIRONMENTS[self.current_env]["search_url"]
    
    def get_project_id(self) -> str:
        """Get the project ID for current environment."""
        return self.ENVIRONMENTS[self.current_env]["project_id"]
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get information about the current environment."""
        config = self.get_config()
        return {
            "environment": self.current_env,
            "name": config["name"],
            "description": config["description"],
            "metadata_url": config["metadata_url"],
            "search_url": config["search_url"]
        }
    
    def list_environments(self) -> Dict[str, Dict[str, str]]:
        """List all available environments."""
        return {
            env: {
                "name": config["name"],
                "description": config["description"]
            }
            for env, config in self.ENVIRONMENTS.items()
        }
    
    def set_custom_environment(self, 
                              metadata_url: str, 
                              search_url: str, 
                              project_id: str = "26851",
                              name: str = "Custom Environment"):
        """
        Add a custom environment configuration.
        
        Args:
            metadata_url: Custom metadata API URL
            search_url: Custom search API URL  
            project_id: Project ID (default: 26851)
            name: Environment display name
        """
        self.ENVIRONMENTS["CUSTOM"] = {
            "name": name,
            "metadata_url": metadata_url,
            "search_url": search_url,
            "project_id": project_id,
            "description": "Custom environment configuration"
        }
        
        # Switch to custom environment
        self.current_env = "CUSTOM"

# Global configuration instance
config = EnvironmentConfig() 