from settings import *
from objects_for_ray_tracing import Sphere
from lighting import *
import numpy.typing as npt
import math
import numpy as np
from numba import njit


@njit(fastmath=True)
def get_rotation_matrix(x, y, z):
    r_matrix_x = np.array([[1.0, 0.0, 0.0],
                           [0.0, np.cos(x), -np.sin(x)],
                           [0.0, np.sin(x), np.cos(x)]])

    r_matrix_y = np.array([[np.cos(y), 0.0, np.sin(y)],
                           [0.0, 1.0, 0.0],
                           [-np.sin(y), 0.0, np.cos(y)]])

    r_matrix_z = np.array([[np.cos(z), -np.sin(z), 0.0],
                           [np.sin(z), np.cos(z), 0.0],
                           [0.0, 0.0, 1.0]])

    return r_matrix_z @ r_matrix_y @ r_matrix_x


def convert_lights_to_matrix(lights: list[LightAmbient | LightPoint | LightDirectional]) -> npt.ArrayLike:
    result = []
    for light in lights:
        if type(light) == LightAmbient:
            result.append((0, light.intensity, 0.0, 0.0, 0.0))
        elif type(light) == LightDirectional:
            vx, vy, vz = light.direction
            result.append((1, light.intensity, vx, vy, vz))
        elif type(light) == LightPoint:
            px, py, pz = light.position
            result.append((2, light.intensity, px, py, pz))
    return np.array(result, dtype=np.float64)


def convert_spheres_to_matrics(spheres: list[Sphere]) -> npt.ArrayLike:
    result = []
    for sphere in spheres:
        cx, cy, cz = sphere.center
        r, g, b = sphere.color
        result.append((
            cx, cy, cz,
            sphere.radius,
            r, g, b,
            sphere.reflective,
            sphere.specular
        ))
    return np.array(result, dtype=np.float64)


@njit(fastmath=True)
def scale_to_canvas(x: int, y: int) -> npt.NDArray[np.float64]:
    return np.array([x * Vw / WIDTH, y * Vh / HEIGHT, d], dtype=np.float64)



@njit(fastmath=True)
def reflect_ray(ray_vec: npt.ArrayLike, normal_vec: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r = np.asarray(ray_vec)
    n = np.asarray(normal_vec)

    return 2 * n * np.dot(n, r) - r


@njit(fastmath=True)
def compute_lightning(surf_point: npt.ArrayLike,
                      normal_vec: npt.ArrayLike,
                      specular: int,
                      camera_vec: npt.ArrayLike,
                      scene_lights: npt.ArrayLike,
                      scene_spheres) -> float:
    intensity = 0.0
    LIGHT_AMBIENT = 0
    # LIGHT_DIRECTIONAL = 1
    LIGHT_POINT = 2
    EPSILON = 0.001
    for light in scene_lights:
        if light[0] == LIGHT_AMBIENT:
            intensity += light[1]
        else:
            if light[0] == LIGHT_POINT:
                vec_to_light = light[2:5] - surf_point
                t_max = 1
            else:
                vec_to_light = np.ascontiguousarray(light[2:5])
                t_max = float('inf')
            shadow_sphere, shadow_t = closest_intersection(surf_point, vec_to_light, scene_spheres, EPSILON, t_max)
            if shadow_sphere is not None:
                continue

            normal_dot_light = np.dot(normal_vec, vec_to_light)
            if normal_dot_light > 0:
                intensity += (light[1] * normal_dot_light /
                              (np.linalg.norm(normal_vec) * np.linalg.norm(vec_to_light)))
            if specular != 1:
                spec_ray = 2 * normal_vec * np.dot(normal_vec, vec_to_light) - vec_to_light
                spec_dot_ray = np.dot(spec_ray, camera_vec)
                if spec_dot_ray > 0:
                    intensity += light[1] * ((spec_dot_ray / (
                            np.linalg.norm(spec_ray) * np.linalg.norm(camera_vec))) ** specular)

    return intensity


@njit(fastmath=True)
def intersect_ray_sphere(ray_origin: npt.ArrayLike,
                         ray_direction: npt.ArrayLike,
                         sphere_center, sphere_radius) -> tuple[float, float]:  # находим пересечения луча со сферой
    radius = sphere_radius
    center_to_origin = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)  # собираем квадратное уравнение
    b = 2 * np.dot(center_to_origin, ray_direction)
    c = np.dot(center_to_origin, center_to_origin) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return float('inf'), float('inf')
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)
    return t1, t2


@njit(fastmath=True)
def closest_intersection(ray_origin: npt.ArrayLike,
                         ray_direction: npt.ArrayLike,
                         spheres,
                         t_min, t_max) -> tuple[npt.NDArray | None, float,]:
    closest_t = float('inf')
    closest_sphere = None
    for sphere in spheres:
        t1, t2 = intersect_ray_sphere(ray_origin, ray_direction, np.ascontiguousarray(sphere[0:3]), sphere[3])
        if t_min <= t1 <= t_max and t1 < closest_t:
            closest_t = t1
            closest_sphere = sphere
        if t_min <= t2 <= t_max and t2 < closest_t:
            closest_t = t2
            closest_sphere = sphere
    return closest_sphere, closest_t


@njit(fastmath=True)
def trace_ray(ray_origin: npt.ArrayLike,
              ray_direction: npt.ArrayLike, spheres: npt.ArrayLike, lights: npt.ArrayLike,
              t_min: float, t_max: float,
              recursion_depth: int
              ):
    closest_sphere, closest_t = closest_intersection(ray_origin,
                                                     ray_direction,
                                                     spheres, t_min, t_max)
    if closest_sphere is None:
        return BACKGROUND_COLOR

    point_vec = ray_origin + (closest_t * ray_direction)
    normal_vec = point_vec - closest_sphere[0:3]
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    sph_r, sph_g, sph_b = closest_sphere[4:7]

    local_lightning = compute_lightning(point_vec, normal_vec, closest_sphere[8], -ray_direction, lights, spheres)

    local_color = (sph_r * local_lightning, sph_g * local_lightning, sph_b * local_lightning)

    reflective = closest_sphere[7]
    if recursion_depth <= 0 or reflective <= 0:
        return local_color

    R = reflect_ray(-ray_direction, normal_vec)
    reflected_color = trace_ray(point_vec, R, spheres, lights, 0.001, float('inf'), recursion_depth - 1)
    r = local_color[0] * (1 - reflective) + reflected_color[0] * reflective
    g = local_color[1] * (1 - reflective) + reflected_color[1] * reflective
    b = local_color[2] * (1 - reflective) + reflected_color[2] * reflective

    return r, g, b
