
// Some general structure inspired by
// https://github.com/spite/THREE.MeshLine/blob/master/src/THREE.MeshLine.js

// Lighting logic from here:
// https://csantosbh.wordpress.com/2014/01/09/custom-shaders-with-three-js-uniforms-textures-and-lighting/

;(function() {
    
    "use strict";
    
    var root = this;
    
    var has_require = typeof require !== 'undefined';
    
    var THREE = root.THREE || has_require && require('three');
    if( !THREE ) {
        throw new Error( 'THREE.contourist requires three.js' );
    }

    // Test if samples have value in range or crossing range.
    var inrange = function(limits, samples) {
        var low_limit = limits[0];
        var high_limit = limits[1];
        var below = false;
        var above = false;
        for (var i=0; i<samples.length; i++) {
            var si = samples[i];
            if (si < low_limit) {
                below = true;
            } else {
                if (si > high_limit) {
                    above = true;
                } else {
                    return true;  // shortcut: point in range
                }
            }
        }
        return (below && above);  // samples cross limits.
    }
    
    var extremum = function(numArray, mnmx) {
        var result = numArray[0];  // apply blows out the stack.
        for (var i=1; i<numArray.length; i++) {
            result = mnmx(result, numArray[i]);
        }
        return result;
    };

    var default_vector3 = function(x, a, b, c) {
        if (x) {
            if (!Array.isArray(x)) {
                return x;
            }
            a = x[0]; b = x[1]; c = x[2];
        }
        return new THREE.Vector3(a, b, c);
    }

    var Irregular3D_Declarations = `
    uniform float f0;

    attribute vec3 A;
    attribute vec3 B;
    attribute vec3 C;
    attribute vec3 D;
    attribute vec4 fABCD;
    attribute float triangle;
    attribute float point_index;
    `;
    
    var Regular3D_Declarations = `
    uniform float f0;
    uniform vec3 origin, u_dir, v_dir, w_dir;

    attribute vec3 ijk, indices;
    attribute vec4 fbuffer_w0, fbuffer_w1;
    `; 

    var Irregular3D_Core = `
    visible = 0.0;  // default to invisible.
    vec3 override_vertex = vec3(0, 0, 0);  // degenerate default.
    vec3 override_normal = vec3(0, 1, 0);  // arbitrary default.
    // FOR DEBUG ONLY!
    /*
    if (point_index < 0.5) {
        override_vertex = A;
    } else {
        if (point_index < 1.5) {
            override_vertex = B;
        } else {
            if (triangle < 0.5) {
                override_vertex = C;
            } else {
                override_vertex = D;
            }
        }
    }
    if (triangle > 0.5) {
        override_normal = vec3(1, 0, 0);
    } */
    //visible = 100000.0;
    // END OF DEBUG
    
    //float triangle = position[0];  // 0 or 1
    //float point_index = position[1]; // 0, 1, or 2
    // bubble sort ABCD on fABCD values
    mat4 sorter;
    sorter[0] = vec4(A, fABCD[0]);
    sorter[1] = vec4(B, fABCD[1]);
    sorter[2] = vec4(C, fABCD[2]);
    sorter[3] = vec4(D, fABCD[3]);
    vec4 save;
    for (int iter=0; iter<3; iter++) {
        for (int i=0; i<3; i++) {
            if (sorter[i][3] > sorter[i+1][3]) {
                save = vec4(sorter[i]);
                sorter[i] = sorter[i+1];
                sorter[i+1] = save;
            }
        }
    }
    // unpack sorted data
    vec3 AA = vec3(sorter[0][0], sorter[0][1], sorter[0][2]);
    float fA = sorter[0][3];
    vec3 BB = vec3(sorter[1][0], sorter[1][1], sorter[1][2]);
    float fB = sorter[1][3];
    vec3 CC = vec3(sorter[2][0], sorter[2][1], sorter[2][2]);
    float fC = sorter[2][3];
    vec3 DD = vec3(sorter[3][0], sorter[3][1], sorter[3][2]);
    float fD = sorter[3][3];
    // find triangle vertices
    vec3 positive_direction = DD - AA;
    vec3 p1 = vec3(0, 0, 0);
    vec3 p2 = vec3(0, 0, 0);
    vec3 p3 = vec3(0, 0, 0);
    bool valid_triangle = false;
    bool two_triangles = false;
    if (fA < f0 && fD > f0) {
        // one or two triangles inside tetrahedron (A,B,C,D)
        if (fB < f0) {
            if (fC < f0) {
                // one triangle (DA, DB, DC)
                if (triangle < 0.5) {
                    valid_triangle = true;
                    interpolate0(DD, fD, AA, fA, f0, 0.0, p2);
                    interpolate0(DD, fD, BB, fB, f0, 0.0, p1);
                    interpolate0(DD, fD, CC, fC, f0, 0.0, p3);
                }
            } else {
                // two vertices on each side of level set:
                // two triangles (AD, AC, BC), (AD, BD, BC)
                two_triangles = true;
                if (triangle < 0.5) {
                    valid_triangle = true;
                    interpolate0(AA, fA, DD, fD, f0, 0.0, p1);
                    interpolate0(AA, fA, CC, fC, f0, 0.0, p2);
                    interpolate0(BB, fB, CC, fC, f0, 0.0, p3);
                } else {
                    valid_triangle = true;
                    interpolate0(AA, fA, DD, fD, f0, 0.0, p1);
                    interpolate0(BB, fB, DD, fD, f0, 0.0, p2);
                    interpolate0(BB, fB, CC, fC, f0, 0.0, p3);
                }
            }
        } else {
            // one triangle (AB, AC, AD)
            if (triangle < 0.5) {
                valid_triangle = true;
                interpolate0(AA, fA, BB, fB, f0, 0.0, p1);
                interpolate0(AA, fA, CC, fC, f0, 0.0, p2);
                interpolate0(AA, fA, DD, fD, f0, 0.0, p3);
            }
        }
    }
    // XXX debugging
    //if (triangle < 0.5) {
    //    valid_triangle = false;
    //}
    if (valid_triangle) {
        if (two_triangles || (triangle < 0.5)) {
            visible = 1.0;
        }
        if (point_index < 0.5) {
            override_vertex = p1;
        } else {
            if (point_index > 1.5) {
                override_vertex = p3;
            } else {
                override_vertex = p2;
            }
        }
        // compute normal
        vec3 test_normal = normalize(cross(p2-p1, p3-p1));
        if (dot(test_normal, positive_direction) < 0.0) {
            override_normal = - test_normal;
        } else {
            override_normal = test_normal;
        }
        //override_normal = vec3(1, 0, 0); // XXXX DEBUG
        //if (triangle < 0.5) {
        //    override_normal = - override_normal;
        //}
    }
    `;

    var Regular3D_Core = `
    float f_origin = fbuffer_w0[0];
    float f_v = fbuffer_w0[1];
    float f_u = fbuffer_w0[2];
    float f_uv = fbuffer_w0[3];
    float f_w = fbuffer_w1[0];
    float f_vw = fbuffer_w1[1];
    float f_uw = fbuffer_w1[2];
    float f_uvw = fbuffer_w1[3];
    
    float i = ijk[0];
    float j = ijk[1];
    float k = ijk[2];
    
    // indices: vertex, triangle, tetra
    float tetrahedron = indices[2];
    float triangle = indices[1];
    float point_index = indices[0];
    
    vec3 A = origin + (i * u_dir) + (j * v_dir) + (k * w_dir);
    float f_A = f_origin;
    vec3 D = A + u_dir + v_dir + w_dir;
    float f_D = f_uvw;
    
    vec3 B, C;
    float f_B, f_C;
    if ((tetrahedron > -0.5) && (tetrahedron < 0.5)) {
        B = A + w_dir;
        f_B = f_w;
        C = A + v_dir + w_dir;
        f_C = f_vw;
    }
    if ((tetrahedron > 0.5) && (tetrahedron < 1.5)) {
        B = A + v_dir;
        f_B = f_v;
        C = A + v_dir + w_dir;
        f_C = f_vw;
    }
    if ((tetrahedron > 1.5) && (tetrahedron < 2.5)) {
        B = A + w_dir;
        f_B = f_w;
        C = A + u_dir + w_dir;
        f_C = f_uw;
    }
    if ((tetrahedron > 2.5) && (tetrahedron < 3.5)) {
        B = A + u_dir;
        f_B = f_u;
        C = A + u_dir + w_dir;
        f_C = f_uw;
    }
    if ((tetrahedron > 3.5) && (tetrahedron < 4.5)) {
        B = A + v_dir;
        f_B = f_v;
        C = A + u_dir + v_dir;
        f_C = f_uv;
    }
    if ((tetrahedron > 4.5) && (tetrahedron < 5.5)) {
        B = A + u_dir;
        f_B = f_u;
        C = A + u_dir + v_dir;
        f_C = f_uv;
    }
    vec4 fABCD = vec4(f_A, f_B, f_C, f_D);
    ` + Irregular3D_Core;  // XXXXX

    var shader_program_start = 'void main() {';

    // added_ihitial calculations should define vec3's override_vertex and override_normal
    //   and also set visible to 1.0 or 0.0.
    var patch_vertex_shader = function(
        source_shader,
        added_global_declarations,
        added_initial_calculations
    ) {
        if (!source_shader.includes(shader_program_start)) {
            throw new Error("Cannot find program start in source shader for patching.");
        }
        var initial_patch = [
            added_global_declarations, 
            "varying float visible;  // contourist visibility flag.",
            interpolate0code,
            shader_program_start, 
            added_initial_calculations
        ].join("\n");
        var patched_shader = source_shader.replace(shader_program_start, initial_patch);
        var begin_vertex = '#include <begin_vertex>';
        if (!patched_shader.includes(begin_vertex)) {
            throw new Error("Cannot find begin vertex in source shader for patching.");
        }
        var vertex_patch = [
            begin_vertex,
            "transformed = vec3(override_vertex); // contourist override"
        ].join("\n");
        patched_shader = patched_shader.replace(begin_vertex, vertex_patch);
        var beginnormal_vertex = '#include <beginnormal_vertex>';
        if (patched_shader.includes(beginnormal_vertex)) {
            var normal_patch = [
                beginnormal_vertex,
                "objectNormal = vec3(override_normal); // contourist override"
            ].join("\n");
            patched_shader = patched_shader.replace(beginnormal_vertex, normal_patch);
        }
        return patched_shader;
    };

    var visibility_patch = `
    varying float visible;

    void main() {
        if (visible<1.0) {
            discard;
        }`;
    var patch_fragment_shader = function(source_shader) {
        if (!source_shader.includes(shader_program_start)) {
            throw new Error("Cannot find program start in fragment shader for patching.");
        }
        var patched_shader = source_shader.replace(shader_program_start, visibility_patch);
        // force front facing to always be true
        var normal_fragment = THREE.ShaderChunk["normal_fragment"];
        normal_fragment = normal_fragment.replace("gl_FrontFacing", "true");
        patched_shader = patched_shader.replace("#include <normal_fragment>", normal_fragment);
        return patched_shader;
    };

    var interpolate0code = `
    bool interpolate0(in vec3 P1, in float fP1, in vec3 P2, in float fP2, 
        in float f0, in float delta, out vec3 interpolated) {
        interpolated = vec3(0, 0, 0);  // degenerate default
        bool use_delta = (abs(delta) > 1e-10);
        float f1scaled = (fP1 - f0);
        float f2scaled = (fP2 - f0);
        if (use_delta) {
            f1scaled = f1scaled / delta;
            f2scaled = f2scaled / delta;
            if (f1scaled > f2scaled) {
                float fsave = f1scaled;
                f1scaled = f2scaled;
                f2scaled = fsave;
                vec3 Psave = P1;
                P1 = P2;
                P2 = Psave;
            }
            float f1ceil = ceil(f1scaled);
            f1scaled = f1scaled - f1ceil;
            f2scaled = f2scaled - f1ceil;
        }
        if ((f1scaled != f2scaled) && (f1scaled * f2scaled < 0.0)) {
            float ratio = f1scaled / (f1scaled - f2scaled);
            interpolated = (ratio * P2) + ((1.0 - ratio) * P1);
        } else {
            return false;
        }
        return true;
    }
    `

    var Irregular2D_Declarations = `
    uniform float f0;
    uniform float delta;

    attribute vec3 A;
    attribute vec3 B;
    attribute vec3 C;
    attribute vec3 f;

    attribute float point_index;

    //varying vec3 vColor;
    varying float visible;
    `;

    var Regular2D_Declarations = `
    uniform float f0;
    uniform float delta;
    uniform vec3 u;
    uniform vec3 v;
    uniform vec3 origin;

    attribute vec2 indices;
    // attribute vec3 position
    attribute vec2 ij;
    attribute vec4 fbuffer;

    //varying vec3 vColor;
    varying float visible;
    `;

    var Regular_Special = `
    float i = ij[0];
    float j = ij[1];
    float point_index = indices[0];
    float triangle_index = indices[1];
    vec3 A = origin + i * u + j * v;
    vec3 C = A + u + v;
    vec3 B;
    float ll = fbuffer[0];
    float lr = fbuffer[1];
    float ul = fbuffer[2];
    float ur = fbuffer[3];
    vec3 f;
    if (triangle_index < 0.5) {
        B = A + v;
        f = vec3(ll, lr, ur);
    } else {
        B = A + u;
        f = vec3(ll, ul, ur);
    }
    `;

    var Irregular2D_Core = `
    float fA0 = f[0];
    float fB0 = f[1];
    float fC0 = f[2];

    vec3 p1;
    vec3 p2;
    bool p2set = false;
    bool p1set = interpolate0(A, fA0, B, fB0, f0, delta, p1);
    if (!p1set) {
        p1set = interpolate0(A, fA0, C, fC0, f0, delta, p1);
    } else {
        p2set = interpolate0(A, fA0, C, fC0, f0, delta, p2);
    }
    if (!p2set) {
        p2set = interpolate0(B, fB0, C, fC0, f0, delta, p2);
    }
    vec3 newPosition = position;  // degenerate default

    if (p1set && (point_index < 0.5)) {
        newPosition = p1;
        //vColor = vec3( 1.0, 0.5, 1.0);
    } else if (p2set) {
        newPosition = p2;
        //vColor = vec3( 1.0, 1.0, 0.5);
    }
    gl_Position = projectionMatrix * modelViewMatrix * vec4( newPosition, 1.0 );
    visible = 0.0;
    if (p1set && p2set) {
        visible = 2.0;
    }`

    var Irregular2D_Vertex_Shader = [
        Irregular2D_Declarations,
        interpolate0code,
        `void main() {`,
        Irregular2D_Core,
        `}`,
        ].join("\n");

    var Regular2D_Vertex_Shader = [
        Regular2D_Declarations,
        interpolate0code,
        `void main() {`,
        Regular_Special,
        Irregular2D_Core,
        `}`,
        ].join("\n");
            
    var Irregular2D_Fragment_Shader = [`
    uniform vec3 color;
    uniform float opacity;
    varying float visible;

    void main() {
        if (visible<1.0) {
            discard;
        }
        gl_FragColor = vec4( color, opacity );

    }
    `].join("\n");

    // coords: rectangular 2d grid of (x, y, z, f(x,y,z))
    //   for point positions and field values.
    var Irregular2D = function(coords, value, delta, limits) {
        
        var nrows = coords.length;
        var ncols = coords[0].length;
        // validate
        for (var i=0; i<nrows; i++) {
            var row = coords[i];
            if (row.length != ncols) {
                throw new Error("all rows must have the same length.");
            }
            for (var j=0; j<ncols; j++) {
                var vector = row[j];
                if (vector.length != 4) {
                    throw new Error("all vector elements shoud have (x, y, z, f)");
                }
            }
        }

        var Abuffer = [];
        var Bbuffer = [];
        var Cbuffer = [];
        var fbuffer = [];

        // per instance
        for (var i=0; i<nrows-1; i++) {
            var rowi = coords[i];
            var rowi1 = coords[i+1];
            for (var j=0; j<ncols-1; j++) {
                var ll = rowi[j];
                var lr = rowi[j+1];
                var ul = rowi1[j];
                var ur = rowi1[j+1];
                //for (var ind=0; ind<2; ind++) {
                    //indices.push(ind);
                var v1 = ll[3]; var v2 = lr[3]; var v3 = ur[3];
                if ((!limits) || (inrange(limits, [v1, v2, v3]))) {
                    Abuffer.push(ll[0], ll[1], ll[2]);
                    Bbuffer.push(lr[0], lr[1], lr[2]);
                    Cbuffer.push(ur[0], ur[1], ur[2]);
                    fbuffer.push(v1, v2, v3)
                }
                    //fbuffer.push(ll[3], lr[3], ur[3])
                //}
                //for (var ind=0; ind<2; ind++) {
                    //indices.push(ind);
                v2 = ul[3];
                if ((!limits) || (inrange(limits, [v1, v2, v3]))) {
                    Abuffer.push(ll[0], ll[1], ll[2]);
                    Bbuffer.push(ul[0], ul[1], ul[2]);
                    Cbuffer.push(ur[0], ur[1], ur[2]);
                    fbuffer.push(ll[3], ul[3], ur[3])
                    //fbuffer.push(ll[3], ul[3], ur[3])
                }
                //}
            }
        }

        return Triangular(Abuffer, Bbuffer, Cbuffer, fbuffer, value, delta);
    };

    var Triangular = function(Abuffer, Bbuffer, Cbuffer, fbuffer, value, delta) {

        var uniforms = {
            color: { type: "c", value: new THREE.Color( 0xffffff ) },
            opacity:   { type: "f", value: 1.0 },
            f0:   { type: "f", value: value },
            delta:   { type: "f", value: delta }
        };

        var setValue = function(value) {
            uniforms.f0.value = value;
        }
        var setDelta = function(value) {
            uniforms.delta.value = value;
        }
        var setOpacity = function(value) {
            uniforms.opacity.value = value;
        }

        var shaderMaterial = new THREE.ShaderMaterial( {
            uniforms:       uniforms,
            vertexShader:   Irregular2D_Vertex_Shader,
            fragmentShader: Irregular2D_Fragment_Shader,
            //blending:       THREE.AdditiveBlending,
            //depthTest:      false,
            transparent:    true
        });

        var buffergeometry = new THREE.InstancedBufferGeometry();

        var indices = [0, 1];

        var mn = extremum(Abuffer, Math.min);
        var mx = extremum(Bbuffer, Math.max);
        var positions = [mn, mn, mn, mx, mx, mx];

        // per mesh
        var point_index = new THREE.Float32BufferAttribute( indices, 1 );
        buffergeometry.addAttribute("point_index", point_index);
        var position_buffer = new THREE.Float32BufferAttribute( positions, 3 );
        buffergeometry.addAttribute("position", position_buffer);

        // per instance
        buffergeometry.addAttribute("A",
            (new THREE.InstancedBufferAttribute( new Float32Array(Abuffer), 3)));
        buffergeometry.addAttribute("B",
            (new THREE.InstancedBufferAttribute( new Float32Array(Bbuffer), 3)));
        buffergeometry.addAttribute("C",
            (new THREE.InstancedBufferAttribute( new Float32Array(Cbuffer), 3)));
        buffergeometry.addAttribute("f",
            (new THREE.InstancedBufferAttribute( new Float32Array(fbuffer), 3)));

        var object = new THREE.LineSegments( buffergeometry, shaderMaterial );

        var result = {
            // encapsulated interface
            object: object,
            geometry: buffergeometry,
            material: shaderMaterial,
            uniforms: uniforms,
            setValue: setValue,
            setDelta: setDelta,
            setOpacity: setOpacity
        };
        return result;
    };
    
    // coords: rectangular 3d grid of (x, y, z, f(x,y,z))
    //   for point positions and field values.
    var Irregular3D = function(coords, value, material, limits) {
        
        var nrows = coords.length;
        var ncols = coords[0].length;
        var ndepth = coords[0][0].length;
        // validate
        for (var i=0; i<nrows; i++) {
            var row = coords[i];
            if (row.length != ncols) {
                throw new Error("all rows must have the same length.");
            }
            for (var j=0; j<ncols; j++) {
                var depth = row[j];
                if (depth.length != ndepth) {
                    throw new Error("all depth lengths should be the same");
                }
                for (var k=0; k<ndepth; k++) {
                    var vector = depth[k];
                    if (vector.length != 4) {
                        throw new Error("all vector elements shoud have (x, y, z, f)");
                    }
                }
            }
        }

        var Abuffer = [];
        var Bbuffer = [];
        var Cbuffer = [];
        var Dbuffer = [];
        var fbuffer = [];

        var Boffsets = [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]]
        var Coffsets = [[0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0]]
        // per instance
        for (var i=0; i<nrows-1; i++) {
            for (var j=0; j<ncols-1; j++) {
                for (var k=0; k<ndepth-1; k++){
                    for (var tetra=0; tetra<6; tetra++) {
                        var Bo = Boffsets[tetra];
                        var Co = Coffsets[tetra];
                        //for (var triangle=0; triangle<2; triangle++) {
                            //for (var ind=0; ind<3; ind++) {
                                // XXX need to do 6 tetrahedra...
                                var Av = coords[i][j][k];
                                var Bv = coords[i+Bo[0]][j+Bo[1]][k+Bo[2]];
                                var Cv = coords[i+Co[0]][j+Co[1]][k+Co[2]];
                                var Dv = coords[i+1][j+1][k+1];
                                //point_indices.push(ind);
                                //triangles.push(triangle);
                                //positions.push(Av[0], Av[1], Av[2]);
                                var v1 = Av[3]; var v2 = Bv[3]; var v3 = Cv[3]; var v4 = Dv[3];
                                if ((!limits) || (inrange(limits, [v1, v2, v3, v4]))) {
                                    Abuffer.push(Av[0], Av[1], Av[2]);
                                    Bbuffer.push(Bv[0], Bv[1], Bv[2]);
                                    Cbuffer.push(Cv[0], Cv[1], Cv[2]);
                                    Dbuffer.push(Dv[0], Dv[1], Dv[2]);
                                    fbuffer.push(v1, v2, v3, v4);
                                    //fbuffer.push(Av[3], Bv[3], Cv[3], Dv[3]);
                                }
                            //}
                        //}
                    }
                }
            }
        }

        return Tetrahedral(Abuffer, Bbuffer, Cbuffer, Dbuffer, fbuffer, material);
    }

    // Tetrahedral vertices with function values at each vertex.
    var Tetrahedral = function(Abuffer, Bbuffer, Cbuffer, Dbuffer, fbuffer, material) {

        if (!material) {
            material = new THREE.MeshNormalMaterial();
        }
        material.side = THREE.DoubleSide;
        var materialShader;
        var uniforms;
        var vertexShader;
        var fragmentShader;
        var result = {};

        material.onBeforeCompile = function ( shader ) {            
            uniforms = shader.uniforms;
            uniforms["f0"] = { type: "f", value: value };
            vertexShader = shader.vertexShader;
            vertexShader = patch_vertex_shader(vertexShader, Irregular3D_Declarations, Irregular3D_Core);
            fragmentShader = shader.fragmentShader;
            fragmentShader = patch_fragment_shader(fragmentShader);
            shader.vertexShader = vertexShader;
            shader.fragmentShader = fragmentShader;
            materialShader = shader;
            result.shader = shader;
            result.vertexShader = vertexShader;
            result.fragmentShader = fragmentShader;
            result.uniforms = uniforms;
        }

        var setValue = function(value) {
            // silently fail if shader is not initialized
            if (uniforms) {
                uniforms.f0.value = value;
            }
        };

        var buffergeometry = new THREE.InstancedBufferGeometry();

        var point_indices = [];
        var triangles = [];
        var positions = [];

        // per mesh
        for (var triangle=0; triangle<2; triangle++) {
            for (var ind=0; ind<3; ind++) {
                point_indices.push(ind);
                triangles.push(triangle);
                positions.push(Abuffer[0], Abuffer[1], Abuffer[2]);  // dummy values
            }
        }
        // per instance
        buffergeometry.addAttribute("A",
            (new THREE.InstancedBufferAttribute( new Float32Array(Abuffer), 3)));
        buffergeometry.addAttribute("B",
            (new THREE.InstancedBufferAttribute( new Float32Array(Bbuffer), 3)));
        buffergeometry.addAttribute("C",
            (new THREE.InstancedBufferAttribute( new Float32Array(Cbuffer), 3)));
        buffergeometry.addAttribute("D",
            (new THREE.InstancedBufferAttribute( new Float32Array(Dbuffer), 3)));
        buffergeometry.addAttribute("fABCD",
            (new THREE.InstancedBufferAttribute( new Float32Array(fbuffer), 4)));
        
        // per mesh
        var triangle_index = new THREE.Float32BufferAttribute( triangles, 1 );
        buffergeometry.addAttribute("triangle", triangle_index);
        var point_index = new THREE.Float32BufferAttribute( point_indices, 1 );
        buffergeometry.addAttribute("point_index", point_index);
        var position_buffer = new THREE.Float32BufferAttribute( positions, 3 );
        buffergeometry.addAttribute("position", position_buffer);

        var object = new THREE.Mesh( buffergeometry, material );

        //var result = {
        // encapsulated interface, and useful debug values :)
        result.object = object;
        result.geometry = buffergeometry,
        result.material = material;
        result.shader = materialShader;
        result.vertexShader = vertexShader;
        result.fragmentShader = fragmentShader;
        result.uniforms = uniforms;
        result.setValue = setValue;
        return result;
    };

    // coords: rectangular grid of f(x,y,z) field values
    //   for point positions on regular grid.
    var Regular2D = function(values, value, delta, origin, u, v, limits) {

        var nrows = values.length;
        var ncols = values[0].length;
        // validate
        for (var i=0; i<nrows; i++) {
            var row = values[i];
            if (row.length != ncols) {
                throw new Error("all rows must have the same length.");
            }
        }
        origin = default_vector3(origin, 0, 0, 0);
        u = default_vector3(u, 1, 0, 0);
        v = default_vector3(v, 0, 1, 0);

        var uniforms = {
            color: { type: "c", value: new THREE.Color( 0xffffff ) },
            opacity:   { type: "f", value: 1.0 },
            f0:   { type: "f", value: value },
            delta:   { type: "f", value: delta },
            u: { type: "v3", value: u },
            v: { type: "v3", value: v },
            origin: { type: "v3", value: origin },
        };

        var setValue = function(value) {
            uniforms.f0.value = value;
        }
        var setDelta = function(value) {
            uniforms.delta.value = value;
        }
        var setOpacity = function(value) {
            uniforms.opacity.value = value;
        }

        var shaderMaterial = new THREE.ShaderMaterial( {
            uniforms:       uniforms,
            vertexShader:   Regular2D_Vertex_Shader,
            fragmentShader: Irregular2D_Fragment_Shader,
            //blending:       THREE.AdditiveBlending,
            //depthTest:      false,
            //transparent:    true
        });

        var buffergeometry = new THREE.InstancedBufferGeometry();
        
        // per mesh
        var indices = [];
        var positions = [];

        // per instance
        var fbuffer = [];
        var ij = []

        for (var i=0; i<nrows-1; i++) {
            var rowi = values[i];
            var rowi1 = values[i+1];
            for (var j=0; j<ncols-1; j++) {
                var ll = rowi[j];
                var lr = rowi[j+1];
                var ul = rowi1[j];
                var ur = rowi1[j+1];
                if ((!limits) || (inrange(limits, [ll, lr, ul, ur]))) {
                    fbuffer.push(ll, lr, ul, ur);
                    ij.push(i, j);
                }
            }
        }
        var e0 = origin.x + u.x + v.x;
        var e1 = origin.y + u.y + v.y;
        var e2 = origin.z + u.z + v.z;
        for (var triangle=0; triangle<2; triangle++) {
            for (var ind=0; ind<2; ind++) {
                indices.push(ind, triangle);
                positions.push(e0, e1, e2); // dummy value
            }
        }
        positions[0] = origin.x;
        positions[1] = origin.y;
        positions[2] = origin.z;

        // add per mesh attributes
        var indices_b = new THREE.Float32BufferAttribute( indices, 2 );
        buffergeometry.addAttribute("indices", indices_b);
        var position_b = new THREE.Float32BufferAttribute( positions, 3 );
        buffergeometry.addAttribute("position", position_b);
        // add per instance attribute
        var ij_b = new THREE.InstancedBufferAttribute(new Float32Array(ij), 2 );
        buffergeometry.addAttribute("ij", ij_b);
        var fbuffer_b = new THREE.InstancedBufferAttribute(new Float32Array(fbuffer), 4 );
        buffergeometry.addAttribute("fbuffer", fbuffer_b);

        var object = new THREE.LineSegments( buffergeometry, shaderMaterial );

        var result = {
            // encapsulated interface
            object: object,
            geometry: buffergeometry,
            material: shaderMaterial,
            uniforms: uniforms,
            setValue: setValue,
            setDelta: setDelta,
            setOpacity: setOpacity
        };
        return result;
    };

    var Regular3D = function(values, value, origin, u, v, w, material, limits) {
        var nrows = values.length;
        var ncols = values[0].length;
        var ndepth = values[0][0].length;
        // validate
        for (var i=0; i<nrows; i++) {
            var row = values[i];
            if (row.length != ncols) {
                throw new Error("all rows must have the same length.");
            }
            for (var j=0; j<ncols; j++) {
                var depth = row[j];
                if (depth.length != ndepth) {
                    throw new Error("all depths must have the same length.");
                }
            }
        }
        origin = default_vector3(origin, 0, 0, 0);
        u = default_vector3(u, 1, 0, 0);
        v = default_vector3(v, 0, 1, 0);
        w = default_vector3(w, 0, 0, 1);
        debugger;
        if (!material) {
            material = new THREE.MeshNormalMaterial();
        }
        material.side = THREE.DoubleSide;
        var materialShader;
        var uniforms;
        var vertexShader;
        var fragmentShader;
        var result = {};

        material.onBeforeCompile = function ( shader ) {            
            uniforms = shader.uniforms;
            uniforms["f0"] = { type: "f", value: value };
            uniforms["origin"] = { type: "v3", value: origin };
            uniforms["u_dir"] = { type: "v3", value: u };
            uniforms["v_dir"] = { type: "v3", value: v };
            uniforms["w_dir"] = { type: "v3", value: w };
            vertexShader = shader.vertexShader;
            vertexShader = patch_vertex_shader(vertexShader, Regular3D_Declarations, Regular3D_Core);
            fragmentShader = shader.fragmentShader;
            fragmentShader = patch_fragment_shader(fragmentShader);
            shader.vertexShader = vertexShader;
            shader.fragmentShader = fragmentShader;
            materialShader = shader;
            result.shader = shader;
            result.vertexShader = vertexShader;
            result.fragmentShader = fragmentShader;
            result.uniforms = uniforms;
        }

        var setValue = function(value) {
            // silently fail if shader is not initialized
            if (uniforms) {
                uniforms.f0.value = value;
            }
        };

        var buffergeometry = new THREE.InstancedBufferGeometry();

        // per instance
        var ijk = [];
        // origin, origin+v, origin+u, origin+u+v, ...
        var fbuffer_w0 = [];
        // origin+w, origin+v+w, origin+u+w, origin+u+v+w, ...
        var fbuffer_w1 = [];

        for (var i=0; i<nrows-1; i++) {
            for (var j=0; j<ncols-1; j++) {
                for (var k=0; k<ndepth-1; k++) {
                    var uvw = [];
                    for (var uu=0; uu<2; uu++) {
                        for (var vv=0; vv<2; vv++) {
                            uvw.push(values[i+uu][j+vv][k]);
                            uvw.push(values[i+uu][j+vv][k+1]);
                        }
                    }
                    if ((!limits) || (inrange(limits, uvw))) {
                        fbuffer_w0.push(uvw[0], uvw[2], uvw[4], uvw[6]);
                        fbuffer_w1.push(uvw[1], uvw[3], uvw[5], uvw[7]);
                        ijk.push(i, j, k);
                    }
                }
            }
        }

        var ijk_b = new THREE.InstancedBufferAttribute(new Float32Array(ijk), 3 );
        buffergeometry.addAttribute("ijk", ijk_b);
        var w0_b = new THREE.InstancedBufferAttribute(new Float32Array(fbuffer_w0), 4 );
        buffergeometry.addAttribute("fbuffer_w0", w0_b);
        var w1_b = new THREE.InstancedBufferAttribute(new Float32Array(fbuffer_w1), 4 );
        buffergeometry.addAttribute("fbuffer_w1", w1_b);

        // per mesh
        var indices = [];
        var positions = [];
        for (var tetra=0; tetra<6; tetra++) {
            for (var triangle=0; triangle<2; triangle++) {
                for (var vertex=0; vertex<3; vertex++) {
                    indices.push(vertex, triangle, tetra);
                    positions.push(origin.x, origin.y, origin.z);
                }
            }
        }
        var indices_b = new THREE.Float32BufferAttribute( indices, 3 );
        buffergeometry.addAttribute("indices", indices_b);
        var position_b = new THREE.Float32BufferAttribute( positions, 3 );
        buffergeometry.addAttribute("position", position_b);

        var object = new THREE.Mesh( buffergeometry, material );
        result.object = object;
        result.geometry = buffergeometry;
        result.material = material;
        result.materialShader = materialShader;
        result.vertexShader = vertexShader;
        result.fragmentShader = fragmentShader;
        result.uniforms = uniforms;
        result.setValue = setValue;
        return result;
    };

    // Exported functionality:
    var contourist = {
        Tetrahedral: Tetrahedral,
        Irregular3D: Irregular3D,
        Triangular: Triangular,
        Irregular2D: Irregular2D,
        Regular2D: Regular2D,
        Regular3D: Regular3D
    };

    if( typeof exports !== 'undefined' ) {
        // Use exports if available
        if( typeof module !== 'undefined' && module.exports ) {
            exports = module.exports = { contourist: contourist };
        }
        exports.contourist = contourist;
    }
    else {
        // Attach exports to THREE
        THREE.contourist = contourist;
    }

}).call(this);