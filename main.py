from settings import *
from lighting import LightAmbient, LightPoint, LightDirectional
from objects_for_ray_tracing import Sphere, Scene
import pygame
import numpy as np
from render_ray_tracing import render
from raytracing_math import get_rotation_matrix

scene = Scene()

global_light = LightAmbient(0.2)
point_light = LightPoint(1.1, (2, 1, 0))
directed_light = LightDirectional(0.2, (1, 4, 4))

lights = [global_light, point_light, directed_light]

spheres = [
    Sphere(center=(0, 0, 3), radius=1, color=(255, 123, 255), specular=100, reflective=0.2),
    Sphere(center=(2, 0, 4), radius=1, color=(255, 0, 15), specular=200, reflective=0.4),
    Sphere(center=(-2, 0, 4), radius=1, color=(200, 120, 0), specular=2000, reflective=0.6),
    Sphere(center=(0, -5001, 0), radius=5000, color=(67, 20, 120), specular=1200, reflective=0.5)
]

def main():
    scene.objects = spheres[:]
    scene.lights = lights[:]
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Render ray-tracing')
    camera_pos = CAMERA_POSITION
    cam_angle_x = CAMERA_ANGEL_X
    cam_angle_y = CAMERA_ANGEL_Y
    cam_angle_z = CAMERA_ANGEL_Z

    r_matrix = get_rotation_matrix(cam_angle_x, cam_angle_y, cam_angle_z)
    
    render(screen, scene, camera_pos, r_matrix)
    pygame.display.flip()
    
    running = True
    
    move_speed = 0.5
    rot_speed = np.radians(10)
    
    while running:
        need_render = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                forward = r_matrix @ np.array([0, 0, 1], dtype=np.float64)
                right = r_matrix @ np.array([1, 0, 0], dtype=np.float64)
                up = r_matrix @ np.array([0, 1, 0], dtype=np.float64)
                if event.key == pygame.K_w:
                    camera_pos += forward * move_speed
                    need_render = True
                elif event.key == pygame.K_s:
                    camera_pos -= forward * move_speed
                    need_render = True
                elif event.key == pygame.K_a:
                    camera_pos -= right * move_speed
                    need_render = True
                elif event.key == pygame.K_d:
                    camera_pos += right * move_speed
                    need_render = True
                elif event.key == pygame.K_q:
                    camera_pos -= up * move_speed
                    need_render = True
                elif event.key == pygame.K_e:
                    camera_pos += up * move_speed
                    need_render = True
                elif event.key == pygame.K_UP:
                    cam_angle_x -= rot_speed
                    need_render = True
                elif event.key == pygame.K_DOWN:
                    cam_angle_x += rot_speed
                    need_render = True
                elif event.key == pygame.K_LEFT:
                    cam_angle_y -= rot_speed
                    need_render = True
                elif event.key == pygame.K_RIGHT:
                    cam_angle_y += rot_speed
                    need_render = True
                    
        if need_render:
            r_matrix = get_rotation_matrix(cam_angle_x, cam_angle_y, cam_angle_z)
            render(screen, scene, camera_pos, r_matrix)
            pygame.display.flip()
                
    pygame.quit()

if __name__ == '__main__':
    main()
