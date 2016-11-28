"""
Generators for HTML5 presentations of contours.
"""

load_three = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r70/three.min.js">
</script>
"""

three_html_fullscreen = """
<!DOCTYPE html>

<!--
Based on:
https://github.com/josdirksen/learning-threejs/blob/master/chapter-05/09-basic-3d-geometries-polyhedron.html
-->

<html>
<head>
 <title>%(title)s</title>

    <style>
        body {
            /* set margin to 0 and overflow to hidden, to go fullscreen */
            margin: 0;
            overflow: hidden;
        }
    </style>

%(load_three)s
</head>

<body>

<div id="%(target_div)s">
</div>

<script type="text/javascript">
    function init() {
        var scene = new THREE.Scene();

        // create a camera, which defines where we're looking at.
        var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);

        // create a render and set the size
        var webGLRenderer = new THREE.WebGLRenderer();
        webGLRenderer.setClearColor(new THREE.Color(0xEEEEEE, 1.0));
        webGLRenderer.setSize(window.innerWidth, window.innerHeight);
        webGLRenderer.shadowMapEnabled = true;
        var triangulation = make_triangulation();
        scene.add(triangulation);
        camera.position.x = -30;
        camera.position.y = 40;
        camera.position.z = 50;
        camera.lookAt(new THREE.Vector3(10, 0, 0));
        document.getElementById("%(target_div)s").appendChild(webGLRenderer.domElement);
        var step = 0;

        function render() {
            triangulation.rotation.y = step += 0.01;

            // render using requestAnimationFrame
            requestAnimationFrame(render);
            webGLRenderer.render(scene, camera);
        };
        render();
    };
    window.onload = init;

    function make_triangulation() {
        var vertices = %(vertices)s;
        var indices = %(indices)s;
        var geom = new THREE.Geometry();
        var geomv = geom.vertices;
        for (var i=0; i<vertices.length; i++) {
            var v = vertices[i];
            geomv.push(new THREE.Vector3(v[0], v[1], v[2]));
        }
        var geomf = geom.faces;
        for (var i=0; i<indices.length; i++) {
            var f = indices[i];
            var face = new THREE.Face3(f[0], f[1], f[2])
            //var normal = new THREE.Vector3(0, 0, 1);
            //face.normal.copy( normal );
            geomf.push(face);
        }
        geom.computeFaceNormals();
        geom.computeVertexNormals();
        var triangulation = createMesh(geom);
        return triangulation;
    };

    function createMesh(geom) {

        // assign two materials
        var meshMaterial = new THREE.MeshNormalMaterial();
        meshMaterial.side = THREE.DoubleSide;
        var wireFrameMat = new THREE.MeshBasicMaterial();
        wireFrameMat.wireframe = true;

        // create a multimaterial
        var mesh = THREE.SceneUtils.createMultiMaterialObject(geom, [meshMaterial, wireFrameMat]);

        return mesh;
    };
</script>

</body>

</html>

"""

def grid_html_page(gridcontour, title="3d contour", load_three=load_three):
    (points, triangles) = gridcontour.get_points_and_triangles()
    vertices = "[%s]" % (",\n    ".join(map(str, map(list, points))))
    indices = "[%s]" % (",\n    ".join(map(str, map(list, triangles))))
    D = {}
    D["title"] = title
    D["target_div"] = "THREE_OUTPUT"
    D["vertices"] = vertices
    D["indices"] = indices
    D["load_three"] = load_three
    return three_html_fullscreen % D

def test_sphere():
    import tetrahedral
    from numpy.linalg import norm
    def sphere(x,y,z):
        return norm([x-5,y-5,z-5])
    G = tetrahedral.Grid3DContour(10,10,10,sphere, 6.0, [[(0,0,0), (5,5,5)]])
    #G = tetrahedral.Grid3DContour(5,5,5,sphere, 4.0, [[(0,0,0), (50,50,50)]])
    print grid_html_page(G, title="sphere")

def test_hyperbola():
    import tetrahedral
    from numpy.linalg import norm
    def hyp(x,y,z):
        return x * y * z
    G = tetrahedral.Grid3DContour(50,50,50,hyp, 100.0, [[(0,0,0), (20,20,20)]])
    #G = tetrahedral.Grid3DContour(5,5,5,sphere, 19.0, [[(0,0,0), (20,20,20)]])
    print grid_html_page(G, title="hyperbola")

def test_torus(offset=5):
    import tetrahedral
    import numpy as np
    from numpy.linalg import norm
    c = np.array((offset, 0), dtype=np.float)
    def torus_function(x,y,z):
        alpha = norm((x,y))
        p = np.array((alpha, z))
        return norm(c - p)
    shift = 3 * offset
    def shift_torus(x,y,z):
        return torus_function(x - shift, y - shift, z - shift)
    side = shift * 2
    G = tetrahedral.Grid3DContour(side, side, side,shift_torus, offset/3.0, [[(0,0,0), (offset+shift, shift, shift)]])
    print grid_html_page(G, title="torus")

def test_wave(side=6, scale=0.2, title="wave"):
    import tetrahedral
    import math
    def f(x,y,z):
        return 1.1 + math.sin(((x - side)**2 + (y - side)**2) * scale) - z
    side2 = 2 * side
    G = tetrahedral.Grid3DContour(side2, side2, side2, f, 0, [[(side, side,0), (20,20,20)]])
    print grid_html_page(G)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        typ = sys.argv[1]
        name = "test_" + typ
        fn = globals()[name]
        fn()
    else:
        test_sphere()
