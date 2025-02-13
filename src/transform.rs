use std::collections::HashMap;
use nalgebra::{Vector3, Quaternion};

#[derive(Debug, Clone)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Transform {
    pub fn new(position: Vector3<f32>, rotation: Quaternion<f32>, scale: Vector3<f32>) -> Self {
        Self { position, rotation, scale }
    }

    pub fn identity() -> Self {
        Self {
            position: Vector3::zeros(),
            rotation: Quaternion::identity(),
            scale: Vector3::new(1.0,1.0,1.0),
        }
    }
}

pub struct TransformSystem {
    transforms: HashMap<u32, Transform>,
    parent_ids: HashMap<u32, Option<u32>>, // Keyed by object ID, value is the parent object ID
}

impl TransformSystem {
    pub fn new() -> Self {
        TransformSystem {
            transforms: HashMap::new(),
            parent_ids: HashMap::new(),
        }
    }

    // Check for cyclic dependencies in the hierarchy
    fn check_cycle(&self, object_id: u32, potential_parent_id: u32) -> bool {
        let mut current_id = potential_parent_id;
        while let Some(parent_id) = self.parent_ids.get(&current_id).and_then(|p| *p) {
            if parent_id == object_id {
                return true; // Found a cycle
            }
            current_id = parent_id;
        }
        false
    }

    // Add an object with a parent if no cycle is detected
    pub fn add_object(&mut self, id: u32, transform: Transform, parent_id: Option<u32>) -> Result<(), String> {
        // Check for cycles before adding the parent
        if let Some(parent_id) = parent_id {
            if self.check_cycle(id, parent_id) {
                return Err(format!("Cannot assign object {} as a child of its own descendant.", id));
            }
        }

        self.transforms.insert(id, transform);
        self.parent_ids.insert(id, parent_id);
        Ok(())
    }

    pub fn get_world_transform(&self, id: u32) -> Transform {
        let mut current_id = id;
        let mut current_transform = self.transforms[&current_id].clone();

        while let Some(parent_id) = self.parent_ids.get(&current_id).and_then(|p| *p) {
            let parent_transform = &self.transforms[&parent_id];
            current_transform = Transform {
                position: parent_transform.position + current_transform.position,
                rotation: parent_transform.rotation * current_transform.rotation,
                scale: parent_transform.scale.component_mul(&current_transform.scale),
            };

            current_id = parent_id;
        }

        current_transform
    }
}