config = {
    "agents": {
        "attraction_weight": 1.05,
        "repulsion_weight_agents": 2,
        "repulsion_weight_shepherd": 1,
        "heading_inertia": 0.5,
        "noise_level": 0.05,
        "neighbor_count": 5,
        "shepherd_detection_dist": 65,
        "agent_interaction_dist": 2,
        "angular_noise": 0.3,
        "disp_per_timestep": 1,
        "grazing_probability": 0.05
    },
    "shepherd": {
        "disp_per_timestep": 1.5,
        "angular_noise": 0.3,
        "group_distance_factor": 1.0
    },
    "environment": {
        "n_agents": 10,
        "n_shepherds": 1
    }
}