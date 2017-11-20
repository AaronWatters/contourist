
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

    var Irregular3D_Declarations = `
    uniform float f0;

    attribute vec3 A;
    attribute vec3 B;
    attribute vec3 C;
    attribute vec3 D;
    attribute vec4 fABCD;
    //attribute float triangle; use position[0]
    //attribute float point_index; use position[1]
    `;
    //var Irregular3D_Declarations = ''; // XXXXXX

    var Irregular3D_Core = `
    visible = 0.0;  // default to invisible.
    vec3 override_vertex = vec3(0, 0, 0);  // degenerate default.
    vec3 override_normal = vec3(0, 0, 1);  // arbitrary default.
    `; /*
    float triangle = position[0];  // 0 or 1
    float point_index = position[1]; // 0, 1, or 2
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
    if (fA < f0 && fD > f0) {
        // one or two triangles inside tetrahedron (A,B,C,D)
        if (fB < f0) {
            if (fC < f0) {
                // one triangle (DA, DB, DC)
                if (triangle < 0.5) {
                    valid_triangle = true;
                    interpolate0(DD, fD, AA, fA, f0, 0.0, p1);
                    interpolate0(DD, fD, BB, fB, f0, 0.0, p2);
                    interpolate0(DD, fD, CC, fC, f0, 0.0, p3);
                }
            } else {
                // two vertices on each side of level set:
                // two triangles (AD, AC, BC), (AD, BD, BC)
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
    if (valid_triangle) {
        visible = 1.0;
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
    }
    `;
    */

    //Irregular3D_Core = '';  // XXXXX

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
        var begin_vertex = THREE.ShaderChunk[ "begin_vertex" ];
        if (!patched_shader.includes(begin_vertex)) {
            throw new Error("Cannot find begin vertex in source shader for patching.");
        }
        var vertex_patch = [
            begin_vertex,
            "transformed = vec3(override_vertex); // contourist override"
        ].join("\n");
        patched_shader = patched_shader.replace(begin_vertex, vertex_patch);
        var beginnormal_vertex = THREE.ShaderChunk[ "beginnormal_vertex" ];
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

    attribute vec4 indices;

    //varying vec3 vColor;
    varying float visible;
    `;

    var Regular_Special = `
    float i = indices[0];
    float j = indices[1];
    float point_index = indices[2];
    float triangle_index = indices[3];
    vec3 A = origin + i * u + j * v;
    vec3 C = A + u + v;
    vec3 B;
    if (triangle_index < 0.5) {
        B = A + v;
    } else {
        B = A + u;
    }
    `;

    var Irregular2D_Core = `
    float fA0 = position[0];
    float fB0 = position[1];
    float fC0 = position[2];

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
    vec3 newPosition = vec3(0, 0, 0);  // degenerate default

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
    var Irregular2D = function(coords, value, delta) {
        
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
            //transparent:    true
        });

        var buffergeometry = new THREE.BufferGeometry();

        var indices = [];
        var Abuffer = [];
        var Bbuffer = [];
        var Cbuffer = [];
        var fbuffer = [];

        for (var i=0; i<nrows-1; i++) {
            var rowi = coords[i];
            var rowi1 = coords[i+1];
            for (var j=0; j<ncols-1; j++) {
                var ll = rowi[j];
                var lr = rowi[j+1];
                var ul = rowi1[j];
                var ur = rowi1[j+1];
                for (var ind=0; ind<2; ind++) {
                    indices.push(ind);
                    Abuffer.push(ll[0], ll[1], ll[2]);
                    Bbuffer.push(lr[0], lr[1], lr[2]);
                    Cbuffer.push(ur[0], ur[1], ur[2]);
                    fbuffer.push(ll[3], lr[3], ur[3])
                }
                for (var ind=0; ind<2; ind++) {
                    indices.push(ind);
                    Abuffer.push(ll[0], ll[1], ll[2]);
                    Bbuffer.push(ul[0], ul[1], ul[2]);
                    Cbuffer.push(ur[0], ur[1], ur[2]);
                    fbuffer.push(ll[3], ul[3], ur[3])
                }
            }
        }
        buffergeometry.addAttribute("point_index",
            (new THREE.BufferAttribute( new Float32Array(indices), 1)));
        buffergeometry.addAttribute("A",
            (new THREE.BufferAttribute( new Float32Array(Abuffer), 3)));
        buffergeometry.addAttribute("B",
            (new THREE.BufferAttribute( new Float32Array(Bbuffer), 3)));
        buffergeometry.addAttribute("C",
            (new THREE.BufferAttribute( new Float32Array(Cbuffer), 3)));
        buffergeometry.addAttribute("position",
            (new THREE.BufferAttribute( new Float32Array(fbuffer), 3)));

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
    var Irregular3D = function(coords, value, shaderID) {
        
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

        if (!shaderID) {
            shaderID = "lambert";
        }

        var shaderInfo = THREE.ShaderLib[shaderID]

        var uniforms = THREE.UniformsUtils.clone( shaderInfo.uniforms )
        uniforms["f0"] = { type: "f", value: value };
        var vertexShader = shaderInfo.vertexShader;
        vertexShader = patch_vertex_shader(vertexShader, Irregular3D_Declarations, Irregular3D_Core);
        var fragmentShader = shaderInfo.fragmentShader;
        fragmentShader = patch_fragment_shader(fragmentShader);

        var setValue = function(value) {
            uniforms.f0.value = value;
        };

        var shaderMaterial = new THREE.ShaderMaterial( {
            uniforms:       uniforms,
            vertexShader:   vertexShader,
            fragmentShader: fragmentShader,
            blending:       THREE.AdditiveBlending,
            depthTest:      false,
            lights:         true,
            side: THREE.DoubleSide
        });

        var buffergeometry = new THREE.BufferGeometry();

        var indices = [];
        var Abuffer = [];
        var Bbuffer = [];
        var Cbuffer = [];
        var Dbuffer = [];
        var fbuffer = [];
        
        // debugging...
        for (var triangle=0; triangle<2; triangle++) {
            for (var point_index=0; point_index<3; point_index++) {
                indices.push(triangle, point_index, 0);
                Abuffer.push(0, 0, 0);
                Bbuffer.push(0, 1, 0);
                Cbuffer.push(1, 0, 0);
                Dbuffer.push(0, 0, 1);
                fbuffer.push(0, 1, 1, 1);
            }
        }
/*
        for (var i=0; i<nrows-1; i++) {
            var rowi = coords[i];
            var rowi1 = coords[i+1];
            for (var j=0; j<ncols-1; j++) {
                var ll = rowi[j];
                var lr = rowi[j+1];
                var ul = rowi1[j];
                var ur = rowi1[j+1];
                for (var ind=0; ind<2; ind++) {
                    indices.push(ind);
                    Abuffer.push(ll[0], ll[1], ll[2]);
                    Bbuffer.push(lr[0], lr[1], lr[2]);
                    Cbuffer.push(ur[0], ur[1], ur[2]);
                    fbuffer.push(ll[3], lr[3], ur[3])
                }
                for (var ind=0; ind<2; ind++) {
                    indices.push(ind);
                    Abuffer.push(ll[0], ll[1], ll[2]);
                    Bbuffer.push(ul[0], ul[1], ul[2]);
                    Cbuffer.push(ur[0], ur[1], ur[2]);
                    fbuffer.push(ll[3], ul[3], ur[3])
                }
            }
        } */
        buffergeometry.addAttribute("A",
            (new THREE.BufferAttribute( new Float32Array(Abuffer), 3)));
        buffergeometry.addAttribute("B",
            (new THREE.BufferAttribute( new Float32Array(Bbuffer), 3)));
        buffergeometry.addAttribute("C",
            (new THREE.BufferAttribute( new Float32Array(Cbuffer), 3)));
        buffergeometry.addAttribute("D",
            (new THREE.BufferAttribute( new Float32Array(Dbuffer), 3)));
        buffergeometry.addAttribute("fABCD",
            (new THREE.BufferAttribute( new Float32Array(fbuffer), 4)));
        buffergeometry.addAttribute("position",
            (new THREE.BufferAttribute( new Float32Array(indices), 3)));

        var object = new THREE.Mesh( buffergeometry, shaderMaterial );

        var result = {
            // encapsulated interface
            object: object,
            geometry: buffergeometry,
            material: shaderMaterial,
            uniforms: uniforms,
            setValue: setValue
        };
        return result;
    };

    // coords: rectangular grid of f(x,y,z) field values
    //   for point positions on regular grid.
    var Regular2D = function(values, value, delta, origin, u, v) {

        var nrows = values.length;
        var ncols = values[0].length;
        // validate
        for (var i=0; i<nrows; i++) {
            var row = values[i];
            if (row.length != ncols) {
                throw new Error("all rows must have the same length.");
            }
        }
        var default_vector3 = function(x, a, b, c) {
            if (x) {
                if (!Array.isArray(x)) {
                    return x;
                }
                a = x[0]; b = x[1]; c = x[2];
            }
            return new THREE.Vector3(a, b, c);
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
        debugger;

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

        var buffergeometry = new THREE.BufferGeometry();

        var indices = [];
        var fbuffer = [];

        for (var i=0; i<nrows-1; i++) {
            var rowi = values[i];
            var rowi1 = values[i+1];
            for (var j=0; j<ncols-1; j++) {
                var ll = rowi[j];
                var lr = rowi[j+1];
                var ul = rowi1[j];
                var ur = rowi1[j+1];
                var diags = [lr, ul];
                for (var triangle=0; triangle<2; triangle++) {
                    var diag = diags[triangle];
                    for (var ind=0; ind<2; ind++) {
                        indices.push(i, j, ind, triangle);
                        fbuffer.push(ll, diag, ur);
                    }
                }
            }
        }
        buffergeometry.addAttribute("indices",
            (new THREE.BufferAttribute( new Float32Array(indices), 4)));
        buffergeometry.addAttribute("position",
            (new THREE.BufferAttribute( new Float32Array(fbuffer), 3)));

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
    }

    // Exported functionality:
    var contourist = {
        Irregular3D: Irregular3D,
        Irregular2D: Irregular2D,
        Regular2D: Regular2D
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