import random
import math

import pyglet
from pyglet.gl import *

from sphere_tri import sphere_tri

win = pyglet.window.Window()

faces, vertices, centers = sphere_tri(radius=3, maxlevel=1)
r = lambda: random.random()
colors = [[r(),r(),r()] for _ in faces]
theta = 0

@win.event
def on_draw():
        # Clear buffers
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0, 4.0/3.0, 1, 40);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(4*math.cos(theta), 6, 4*math.sin(theta), 0, 0, 0, 0, 1, 0);

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glBegin(GL_LINES);
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(10, 0, 0);
        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 10, 0);
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 10);
        glEnd();

        for f,c in zip(faces, colors):
                coords = vertices[f[0]], vertices[f[1]], vertices[f[2]]
                glBegin(GL_TRIANGLES)
                glColor3f(c[0], c[1], c[2])
                glVertex3d(coords[0][0], coords[0][1], coords[0][2])
                glVertex3d(coords[1][0], coords[1][1], coords[1][2])
                glVertex3d(coords[2][0], coords[2][1], coords[2][2])
                glEnd()
        global theta 
        theta += 0.1
        print "drawn..", theta

pyglet.clock.schedule_interval(lambda x: win.dispatch_event('on_draw'), 1/30.0)
pyglet.app.run()
