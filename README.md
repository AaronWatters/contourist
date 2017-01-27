# contourist

Data structures, algorithms and display tools for generating 2d contours
and 3d isosurfaces.

# Contour interpolations

[Contour lines](https://en.wikipedia.org/wiki/Contour_line) define
a boundary as a sequence of line segments approximating f(x,y) = v
for some function f, some value v and x and y varying in some specified region.

![Contour display example](contours.png)

# Isosurfaces

[Isosurfaces](https://en.wikipedia.org/wiki/Isosurface) define
a boundary as a collection of oriented triangles approximating f(x,y,z) = v
for some function f, some value v and (x,y,z) varying in some specified volume.

![Isosurface display example](isosurface.png)

Here is a live demo of an isosurface displayed using jsfiddle:
[https://jsfiddle.net/AaronWatters/9qszgyaj/](https://jsfiddle.net/AaronWatters/9qszgyaj/).

# Morphing 3d Isosurfaces

A morphing isosurface represents the point set satisfying f(x,y,z,t) = v
for some function f, some value v and (x,y,z,t) varying in some specified volume
and time extent.  It displays a smoothy evolving isosurface for f(x,y,z,t0) = v
as t0 smoothly transitions from a minimum to a maximum time value.

Here is a live demo of a morphing isosurface displayed using jsfiddle:
[https://jsfiddle.net/AaronWatters/x35crpb0/](https://jsfiddle.net/AaronWatters/x35crpb0/).
