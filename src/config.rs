use crate::types::Config;
use anyhow::Result;
use std::fs;

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}
