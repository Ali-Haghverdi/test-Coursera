import numpy as np
from numpy import linalg as LA
from typing import List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random


def analyze_layup(fiber_coordinates_list: List[np.ndarray],
                  layup: str = "[0]",
                  ply_thickness: float = 0.2,
                  total_plies: int = 1) -> Dict:
    """
    General function for analyzing fiber layups.
    Works for both UD ([0]n) and cross-ply ([0/90]nS) laminates.

    Args:
        fiber_coordinates_list: List of fiber coordinates
        layup: Layup sequence (e.g., "[0]20" or "[0/90]4S")
        ply_thickness: Individual ply thickness in mm
        total_plies: Total number of plies
    """
    print("\nStarting Layup Analysis")
    print("-" * len("Starting Layup Analysis"))
    print(f"Input layup sequence: {layup}")
    print(f"Total number of fibers: {len(fiber_coordinates_list)}")
    print(f"Individual ply thickness: {ply_thickness:.6f} mm")
    print(f"Total number of plies: {total_plies}")
    print(f"Expected total thickness: {ply_thickness * total_plies:.6f} mm")

    # Determine layup type
    is_ud = '[0]' in layup and '/90' not in layup
    print(f"Layup type: {'Unidirectional' if is_ud else 'Cross-ply'}\n")

    # Step 1: Initial Processing and Global Alignment
    print("-" * len("Step 1: Initial Processing and Rotations"))
    print("Step 1: Initial Processing and Rotations")
    print("-" * len("Step 1: Initial Processing and Rotations"))
    print(f"\nThis code uses Spherical Coordinate System where:\n")
    print(f"- Phi is the out-of-plane angle")
    print(f"- Theta is the in-plane angle")
    aligned_data = initial_processing(fiber_coordinates_list)

    # Step 2: Layer Detection
    if not is_ud:
        print("-" * len("Step 2: Progressive Layer Detection"))
        print("\nStep 2: Progressive Layer Detection")
        print("-" * len("Step 2: Progressive Layer Detection"))

        print("\nStep 2.1: Top Layer Identification and X-axis Alignment")
        print("-" * len("Step 2.1: Top Layer Identification and X-axis Alignment"))

        # Detect layers with physical parameters
        layer_data = detect_layers(aligned_data['aligned_coordinates'])

        # Verify layer detection
        verify_layer_detection(layer_data['layers'],
                               expected_thickness=ply_thickness * total_plies,
                               ply_thickness=ply_thickness)

        # Print layer sequence verification
        print("\nLayer Sequence Verification")
        print("-" * len("Layer Sequence Verification"))
        print(f"Expected number of layers: {total_plies}")
        print(f"Detected number of layers: {len(layer_data['layers'])}")
        print(f"Expected sequence: {layup}")

        return {
            'aligned_data': aligned_data,
            'layer_data': layer_data,
            'layup_params': {
                'sequence': layup,
                'ply_thickness': ply_thickness,
                'total_plies': total_plies
            }
        }
    else:
        return {
            'aligned_data': aligned_data,
            'layup_params': {
                'sequence': layup,
                'ply_thickness': ply_thickness,
                'total_plies': total_plies
            }
        }


def initial_processing(fiber_coordinates_list: List[np.ndarray]) -> Dict:
    """
    Convert to spherical coordinates and align laminate to XY plane.
    """
    # Convert each fiber to spherical coordinates
    spherical_data = []
    for fiber in fiber_coordinates_list:
        fiber_data = convert_to_spherical_coords(fiber)
        spherical_data.append(fiber_data)

    # Plot initial angles before any processing
    plot_initial_angles(spherical_data)

    # Calculate alignment angles
    alignment_angles = calculate_alignment_angles(fiber_coordinates_list)
    print("\nRotation angles calculation")
    print("-" * len("Rotation angles calculation"))
    print(f"Actual out-of-plane rotation angle: {alignment_angles['rot2']:.2f}°")
    print(f"Actual in-plane rotation angle: {alignment_angles['rot1']:.2f}°")

    # Apply rotations to align with XY plane
    aligned_coordinates = align_to_xy(fiber_coordinates_list, alignment_angles)

    # Add normalization step
    normalized_coordinates = normalize_coordinates(aligned_coordinates)

    # Verify alignment with normalized coordinates
    verify_alignment(normalized_coordinates, alignment_angles, spherical_data)

    return {
        'spherical_data': spherical_data,
        'aligned_coordinates': normalized_coordinates
    }


def convert_to_spherical_coords(fiber_coordinates: np.ndarray) -> Dict:
    """
    Convert fiber coordinates to spherical coordinates considering all segments.
    """
    # Calculate fiber direction vector (end - start)
    fiber_vector = fiber_coordinates[-1] - fiber_coordinates[0]

    # Calculate spherical coordinates
    phi = np.degrees(np.arccos(fiber_vector[2] / LA.norm(fiber_vector)))
    theta = np.degrees(np.arctan2(fiber_vector[1], fiber_vector[0]))

    return {
        'phi': phi,
        'theta': theta,
        'direction_vector': fiber_vector
    }


def plot_initial_angles(spherical_data: List[Dict]) -> None:
    """
    Plot and print initial phi and theta distributions before any rotations.
    """
    # Extract initial angles directly from spherical_data
    initial_phi = [fiber['phi'] for fiber in spherical_data]
    initial_theta = [fiber['theta'] for fiber in spherical_data]

    # Calculate means
    mean_phi = np.mean(initial_phi)
    mean_theta = np.mean(initial_theta)

    # Print statistics
    print("\nInitial fiber orientations statistics: Raw Data")
    print("-" * len("Initial fiber orientations statistics: Raw Data"))

    print(f"\nPhi_initial (degrees):\n")
    print(f"  Initial_Min: {min(initial_phi):.2f}")
    print(f"  Initial_Max: {max(initial_phi):.2f}")
    print(f"  Initial_Mean: {mean_phi:.2f}")
    print(f"  Initial_Median: {np.median(initial_phi):.2f}")
    print(f"  Initial_Std Dev: {np.std(initial_phi):.2f}")

    print(f"\nTheta_initial (degrees):\n")
    print(f"  Initial_Min: {min(initial_theta):.2f}")
    print(f"  Initial_Max: {max(initial_theta):.2f}")
    print(f"  Initial_Mean: {mean_theta:.2f}")
    print(f"  Initial_Median: {np.median(initial_theta):.2f}")
    print(f"  Initial_Std Dev: {np.std(initial_theta):.2f}")

    # Plot distributions
    plt.figure(figsize=(15, 6))

    # Phi distribution
    plt.subplot(121)
    plt.hist(initial_phi, bins=36, range=(0, 180))
    plt.axvline(x=mean_phi, color='r', linestyle='--',
                label=f'Initial_Mean φ ({mean_phi:.2f}°)')
    plt.title("Initial Phi Distribution")
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Frequency")
    plt.legend()

    # Theta distribution
    plt.subplot(122)
    plt.hist(initial_theta, bins=36, range=(-180, 180))
    plt.axvline(x=mean_theta, color='r', linestyle='--',
                label=f'Initial_Mean θ ({mean_theta:.2f}°)')
    plt.title("Initial Theta Distribution")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()


@dataclass
class PSOParams:
    """Parameters for PSO algorithm"""
    num_particles: int = 50
    max_iter: int = 100
    w: float = 0.7  # inertia weight
    c1: float = 1.5  # cognitive learning rate
    c2: float = 1.5  # social learning rate
    bounds: List = (-45, 45)  # rotation angle bounds
    phi_tolerance: float = 0.1
    theta_tolerance: float = 0.1
    min_cost_tolerance: float = 1e-4


class Particle:
    """Particle class for PSO"""

    def __init__(self, bounds):
        self.position = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(2)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(2)])
        self.best_position = self.position.copy()
        self.best_score = float('inf')


def calculate_angles_after_rotation(fiber_coordinates_list: List[np.ndarray], angles: np.ndarray) -> tuple:
    """
    Calculate mean phi and theta after applying rotations.
    Returns: (mean_phi, mean_theta, all_phi, all_theta)
    """
    rot_y = rotation_matrix('y', angles[0])
    rot_z = rotation_matrix('z', angles[1])
    total_rotation = np.dot(rot_z, rot_y)

    final_phi, final_theta = [], []
    for fiber in fiber_coordinates_list:
        rotated = np.dot(fiber, total_rotation.T)
        vector = rotated[-1] - rotated[0]

        phi = np.degrees(np.arccos(vector[2] / LA.norm(vector)))
        theta = np.degrees(np.arctan2(vector[1], vector[0]))

        final_phi.append(phi)
        final_theta.append(theta)

    return np.mean(final_phi), np.mean(final_theta), np.array(final_phi), np.array(final_theta)


def pso_alignment(fiber_coordinates_list: List[np.ndarray], params: PSOParams, verbose: bool = True) -> Dict:
    """
    PSO implementation for finding optimal rotation angles.
    """

    def cost(angles):
        mean_phi, _, _, _ = calculate_angles_after_rotation(fiber_coordinates_list, angles)
        return (mean_phi - 90) ** 2

    # Initialize particles
    particles = [Particle(params.bounds) for _ in range(params.num_particles)]
    global_best_position = particles[0].position.copy()
    global_best_score = float('inf')

    # Lists for tracking convergence
    convergence_history = []
    angle_history = []

    # Add angle accuracy threshold
    angle_tolerance = 0.001  # degrees

    # Main PSO loop
    for iteration in range(params.max_iter):
        for particle in particles:
            # Calculate current score
            score = cost(particle.position)

            # Update particle's best
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particle.position.copy()

        # Update particles
        for particle in particles:
            # Update velocity
            inertia = params.w * particle.velocity
            cognitive = params.c1 * random.random() * (particle.best_position - particle.position)
            social = params.c2 * random.random() * (global_best_position - particle.position)

            particle.velocity = inertia + cognitive + social

            # Update position with bounds
            particle.position = np.clip(particle.position + particle.velocity, params.bounds[0], params.bounds[1])

        # Track progress
        convergence_history.append(global_best_score)
        angle_history.append(global_best_position.copy())

        # Check if we've reached desired accuracy
        mean_phi, mean_theta, _, _ = calculate_angles_after_rotation(fiber_coordinates_list, global_best_position)
        if abs(mean_phi - 90) < angle_tolerance and abs(mean_theta) < angle_tolerance:
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break

        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{params.max_iter}")
            print(f"Best score: {global_best_score:.6f}")
            print(f"Mean Phi: {mean_phi:.2f}°, Mean Theta: {mean_theta:.2f}°\n")

    # Calculate final angles
    final_mean_phi, final_mean_theta, all_phi, all_theta = calculate_angles_after_rotation(
        fiber_coordinates_list, global_best_position)

    return {
        'rot2': global_best_position[0],  # Y-rotation
        'rot1': global_best_position[1],  # Z-rotation
        'final_mean_phi': final_mean_phi,
        'final_mean_theta': final_mean_theta,
        'convergence_history': convergence_history,
        'angle_history': angle_history
    }


def plot_convergence(history: List[float]) -> None:
    """Plot convergence history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('PSO Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def plot_angle_distributions(phi: np.ndarray, theta: np.ndarray, mean_phi: float, mean_theta: float) -> None:
    """Plot final angle distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Phi distribution
    ax1.hist(phi, bins=36, range=(0, 180))
    ax1.axvline(x=90, color='r', linestyle='--', label='Target φ (90°)')
    ax1.axvline(x=mean_phi, color='g', linestyle='-',
                label=f'Actual mean φ ({mean_phi:.2f}°)')
    ax1.set_title('Final Phi Distribution')
    ax1.set_xlabel('Phi (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Theta distribution
    ax2.hist(theta, bins=36, range=(-180, 180))
    ax2.axvline(x=0, color='r', linestyle='--', label='Target θ (0°)')
    ax2.axvline(x=mean_theta, color='g', linestyle='-',
                label=f'Actual mean θ ({mean_theta:.2f}°)')
    ax2.set_title('Final Theta Distribution')
    ax2.set_xlabel('Theta (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def calculate_alignment_angles(fiber_coordinates_list: List[np.ndarray]) -> Dict:
    """
    Main function to calculate alignment angles using PSO.
    """
    # Initialize PSO parameters
    params = PSOParams()

    # Run PSO
    result = pso_alignment(fiber_coordinates_list, params)

    return {
        'rot2': result['rot2'],
        'rot1': result['rot1']
    }


def align_to_xy(fiber_coordinates_list: List[np.ndarray], alignment_angles: Dict) -> List[Dict]:
    """
    Apply rotations to align laminate with XY plane.
    """
    # Create rotation matrices
    rotation_to_xy = rotation_matrix('z', alignment_angles['rot1'])
    rot_y = rotation_matrix('y', alignment_angles['rot2'])
    total_rotation = np.dot(rotation_to_xy, rot_y)

    aligned_data = []
    for fiber_coordinates in fiber_coordinates_list:
        # Apply rotation directly to original coordinates
        aligned_coords = np.dot(fiber_coordinates, total_rotation.T)

        # Update direction vector
        aligned_vector = aligned_coords[-1] - aligned_coords[0]

        aligned_data.append({
            'coordinates': aligned_coords,
            'vector': aligned_vector,
            'z_mean': np.mean(aligned_coords[:, 2])
        })

    return aligned_data


def normalize_coordinates(aligned_coordinates: List[Dict]) -> List[Dict]:
    """
    Normalize coordinates so Z starts from 0 at bottom of laminate.

    Args:
        aligned_coordinates: List of dictionaries containing fiber coordinates
    Returns:
        List of dictionaries with normalized coordinates
    """
    print("\nNormalizing Coordinates")
    print("-" * len("Normalizing Coordinates"))

    # Store original z-range for verification
    original_z = np.array([np.mean(fiber['coordinates'][:, 2])
                           for fiber in aligned_coordinates])
    print(f"Original Z-range: {np.min(original_z):.4f} to {np.max(original_z):.4f} mm")

    # Find minimum z across all fibers
    min_z = min(np.min(fiber['coordinates'][:, 2]) for fiber in aligned_coordinates)

    # Normalize coordinates
    normalized_coordinates = []
    for fiber in aligned_coordinates:
        normalized_coords = fiber['coordinates'].copy()
        normalized_coords[:, 2] -= min_z

        normalized_coordinates.append({
            'coordinates': normalized_coords,
            'vector': fiber['vector'],
            'z_mean': np.mean(normalized_coords[:, 2])
        })

    # Verify normalization
    new_z = np.array([fiber['z_mean'] for fiber in normalized_coordinates])
    print(f"Normalized Z-range: {np.min(new_z):.4f} to {np.max(new_z):.4f} mm")

    # Visualize before/after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Before normalization
    ax1.hist(original_z, bins=50)
    ax1.set_title('Z Distribution Before Normalization')
    ax1.set_xlabel('Z position (mm)')
    ax1.set_ylabel('Count')
    ax1.grid(True)

    # After normalization
    ax2.hist(new_z, bins=50)
    ax2.set_title('Z Distribution After Normalization')
    ax2.set_xlabel('Z position (mm)')
    ax2.set_ylabel('Count')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return normalized_coordinates


def verify_alignment(aligned_data: List[Dict], alignment_angles: Dict, spherical_data: List[Dict]) -> None:
    """
    Verify proper alignment with XY plane and analyze final fiber orientations.
    """
    # Calculate vectors using the stored coordinates
    vectors = np.array([data['vector'] for data in aligned_data])
    avg_vector = np.mean(vectors, axis=0)

    # Calculate final phi and final theta for each fiber
    final_phi, final_theta = [], []
    for fiber in aligned_data:
        vector = fiber['vector']

        # Calculate angles
        phi = np.degrees(np.arccos(vector[2] / LA.norm(vector)))
        theta = np.degrees(np.arctan2(vector[1], vector[0]))

        final_phi.append(phi)
        final_theta.append(theta)

    # Calculate means
    mean_phi = np.mean(final_phi)
    mean_theta = np.mean(final_theta)

    # Print statistics
    print("\nFinal fiber orientations statistics: After XY-plane alignment")
    print("-" * len("Final fiber orientations statistics: After XY-plane alignment"))

    print(f"\nPhi_Final (degrees):\n")
    print(f"  Final_Min: {np.min(final_phi):.2f}")
    print(f"  Final_Max: {np.max(final_phi):.2f}")
    print(f"  Final_Mean: {np.mean(final_phi):.2f}")
    print(f"  Final_Median: {np.median(final_phi):.2f}")
    print(f"  Final_Std Dev: {np.std(final_phi):.2f}")

    print(f"\nTheta_Final (degrees):\n")
    print(f"  Final_Min: {np.min(final_theta):.2f}")
    print(f"  Final_Max: {np.max(final_theta):.2f}")
    print(f"  Final_Mean: {np.mean(final_theta):.2f}")
    print(f"  Final_Median: {np.median(final_theta):.2f}")
    print(f"  Final_Std Dev: {np.std(final_theta):.2f}")

    # Print alignment accuracy
    print(f"\nExpected average Phi after rotations: {90:.2f}°")
    print(f"Expected average Theta after rotations: {0:.2f}°")

    print(f"\nActual average Phi after rotations: {mean_phi:.2f}°")
    print(f"Actual average Theta after rotations: {mean_theta:.2f}°")

    print("\nFinal Angles Mean Squared Error (MSE)")
    print("-" * len("Mean Angles Mean Squared Error (MSE)"))
    print(f"Average Phi MSE: {calculate_angle_error(90, mean_phi):.6f}")
    print(f"Average Theta MSE: {calculate_angle_error(0, mean_theta):.6f}")

    # Plot distributions with both expected and actual means
    plt.figure(figsize=(15, 6))

    # Phi distribution
    plt.subplot(121)
    plt.hist(final_phi, bins=36, range=(0, 180))
    plt.axvline(x=90, color='r', linestyle='--', label='Expected mean φ (90°)')
    plt.axvline(x=mean_phi, color='g', linestyle='-',
                label=f'Actual mean φ ({mean_phi:.2f}°)')
    plt.title("Final Phi Distribution")
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Frequency")
    plt.legend()

    # Theta distribution
    plt.subplot(122)
    plt.hist(final_theta, bins=36, range=(-180, 180))
    plt.axvline(x=0, color='r', linestyle='--', label='Expected mean θ (0°)')
    plt.axvline(x=mean_theta, color='g', linestyle='-',
                label=f'Actual mean θ ({mean_theta:.2f}°)')
    plt.title("Final Theta Distribution")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()


def detect_layers(aligned_coordinates: List[Dict]) -> Dict:
    """
    Main function for layer detection after XY alignment.
    """
    # Step 2.1: Find and align top layer
    top_layer = identify_top_layer(aligned_coordinates)
    all_coordinates_aligned = align_to_xaxis(aligned_coordinates, top_layer)

    # Step 2.2: Detect remaining layers
    layers = detect_layer_sequence(all_coordinates_aligned)

    return {
        'top_layer': top_layer,
        'layers': layers
    }


def identify_top_layer(aligned_coordinates: List[Dict]) -> Dict:
    """
    Identify top layer using z-coordinates and orientation analysis.
    """
    print("\nStarting Top Layer Identification")
    print("-" * len("Starting Top Layer Identification"))

    # Extract z-coordinates and theta values
    z_coords = np.array([fiber['z_mean'] for fiber in aligned_coordinates])
    theta_values = np.array([np.degrees(np.arctan2(fiber['vector'][1],
                                                   fiber['vector'][0]))
                             for fiber in aligned_coordinates])

    # Step 1: Z-coordinate Analysis
    print("\nZ-coordinate Statistics:")
    print(f"Min Z: {np.min(z_coords):.4f} mm")
    print(f"Max Z: {np.max(z_coords):.4f} mm")
    print(f"Mean Z: {np.mean(z_coords):.4f} mm")
    print(f"Std Z: {np.std(z_coords):.4f} mm")

    # Visualize Z-distribution
    plt.figure(figsize=(10, 6))
    plt.hist(z_coords, bins=50, density=True)
    plt.title('Z-coordinate Distribution')
    plt.xlabel('Z position (mm)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

    # Step 2: Top Layer Identification
    z_threshold = np.percentile(z_coords, 95)
    top_indices = np.where(z_coords >= z_threshold)[0]

    print(f"\nTop Layer Identification:")
    print(f"Z-threshold: {z_threshold:.4f} mm")
    print(f"Number of fibers in top layer: {len(top_indices)}")
    print(f"Percentage of total fibers: {len(top_indices) / len(z_coords) * 100:.2f}%")

    # Step 3: Orientation Analysis of Top Layer
    top_theta = theta_values[top_indices]

    # Visualize top layer orientation
    plt.figure(figsize=(10, 6))
    plt.hist(top_theta, bins=36, range=(-180, 180))
    plt.title('Theta Distribution in Top Layer')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Count')
    plt.axvline(x=np.mean(top_theta), color='r',
                label=f'Mean θ: {np.mean(top_theta):.2f}°')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate statistics
    mean_theta = np.mean(top_theta)
    std_theta = np.std(top_theta)

    print(f"\nTop Layer Orientation Statistics:")
    print(f"Mean Theta: {mean_theta:.2f}°")
    print(f"Std Theta: {std_theta:.2f}°")
    print(f"Min Theta: {np.min(top_theta):.2f}°")
    print(f"Max Theta: {np.max(top_theta):.2f}°")

    return {
        'indices': top_indices,
        'mean_theta': mean_theta,
        'std_theta': std_theta,
        'z_range': (z_threshold, np.max(z_coords))
    }


def align_to_xaxis(aligned_coordinates: List[Dict],
                   top_layer: Dict) -> List[Dict]:
    """
    Rotate all coordinates to align top layer with X-axis.
    """
    # Calculate required rotation
    rotation_angle = -top_layer['mean_theta']
    rotation_matrix_z = rotation_matrix('z', rotation_angle)

    # Apply rotation to all fibers
    aligned_data = []
    for fiber in aligned_coordinates:
        # Rotate coordinates
        rotated_coords = np.dot(fiber['coordinates'], rotation_matrix_z.T)
        # Update vector
        rotated_vector = rotated_coords[-1] - rotated_coords[0]

        aligned_data.append({
            'coordinates': rotated_coords,
            'vector': rotated_vector,
            'z_mean': np.mean(rotated_coords[:, 2])
        })

    return aligned_data


def detect_layer_sequence(aligned_coordinates: List[Dict]) -> List[Dict]:
    """
    Detect layers progressively through thickness.
    """
    # Sort by z-coordinate
    z_coords = np.array([fiber['z_mean'] for fiber in aligned_coordinates])
    sorted_indices = np.argsort(z_coords)[::-1]  # Top to bottom

    # Initialize layer detection
    layers = []
    current_orientation = 0  # Start with 0° (top layer)

    # Parameters for layer detection
    window_size = len(aligned_coordinates) // 16  # Approximate layer size
    z_values = np.sort(z_coords)[::-1]

    # Moving window analysis
    for i in range(0, len(z_values), window_size):
        window_indices = sorted_indices[i:i + window_size]

        # Analyze window orientation
        theta_values = np.array([np.degrees(np.arctan2(
            aligned_coordinates[idx]['vector'][1],
            aligned_coordinates[idx]['vector'][0]))
            for idx in window_indices])

        # Calculate layer statistics
        mean_theta = np.mean(theta_values)
        std_theta = np.std(theta_values)

        # Classify layer
        layer_data = {
            'indices': window_indices,
            'orientation': current_orientation,
            'z_range': (np.min(z_coords[window_indices]),
                        np.max(z_coords[window_indices])),
            'statistics': {
                'mean_theta': mean_theta,
                'std_theta': std_theta,
                'fiber_count': len(window_indices)
            }
        }

        layers.append(layer_data)
        current_orientation = 90 if current_orientation == 0 else 0

    return layers


def verify_layer_detection(layers: List[Dict], expected_thickness: float, ply_thickness: float) -> None:
    """
    Verify layer detection results.
    """
    # Plot layer positions and orientations
    plt.figure(figsize=(12, 6))

    for i, layer in enumerate(layers):
        z_mid = np.mean(layer['z_range'])
        plt.scatter(layer['statistics']['mean_theta'], z_mid,
                    label=f'Layer {i + 1}')

    plt.axvline(x=0, color='r', linestyle='--', label='0°')
    plt.axvline(x=90, color='g', linestyle='--', label='90°')

    plt.xlabel('Mean Theta (degrees)')
    plt.ylabel('Z position (mm)')
    plt.title('Layer Detection Results')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print layer statistics
    print("\nLayer Detection Summary")
    print("-" * len("Layer Detection Summary"))
    for i, layer in enumerate(layers):
        print(f"\nLayer {i + 1}:")
        print(f"Orientation: {layer['orientation']}°")
        print(f"Z-range: {layer['z_range'][0]:.3f} to {layer['z_range'][1]:.3f} mm")
        print(f"Thickness: {layer['z_range'][1] - layer['z_range'][0]:.3f} mm")
        print(f"Mean θ: {layer['statistics']['mean_theta']:.2f}°")
        print(f"Std θ: {layer['statistics']['std_theta']:.2f}°")
        print(f"Fiber count: {layer['statistics']['fiber_count']}")


def rotation_matrix(axis, angle):
    angle = np.radians(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


def calculate_z_average(fiber_coordinates):
    z_values = fiber_coordinates[:, 2]
    return (np.min(z_values) + np.max(z_values)) / 2


def calculate_angle_error(expected: float, actual: float) -> float:
    """
    Calculate angle error using MSE.
    """
    return (expected - actual) ** 2
