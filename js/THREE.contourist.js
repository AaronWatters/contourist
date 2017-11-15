
// Some general structure inspired by
// https://github.com/spite/THREE.MeshLine/blob/master/src/THREE.MeshLine.js
debugger;

;(function() {
    
    "use strict";
    
    var root = this;
    
    var has_require = typeof require !== 'undefined';
    
    var THREE = root.THREE || has_require && require('three');
    if( !THREE ) {
        throw new Error( 'THREE.contourist requires three.js' );
    }

    var Irregular2D_Vertex_Shader = [`
    uniform float f0;
    uniform float delta;

    attribute vec3 A;
    attribute vec3 B;
    attribute vec3 C;

    attribute float point_index;

    //varying vec3 vColor;
    varying float visible;

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

    void main() {

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
        }
    }
    `].join("\n");

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

    // coordinates_and_values: rectangular grid of (x, y, z, f(x,y,z))
    //   for point positions and field values.
    var Irregular2D = function(coordinates_and_values, value, delta) {

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
            blending:       THREE.AdditiveBlending,
            depthTest:      false,
            transparent:    true
        });

        var buffergeometry = new THREE.BufferGeometry();

        var indices = [];
        var Abuffer = [];
        var Bbuffer = [];
        var Cbuffer = [];
        var fbuffer = [];
        for (var i = 0; i<6; i++) {
            indices.push(i % 2);
            Abuffer.push(0, 0, 0);
            Bbuffer.push(1, 2, 0);
            Cbuffer.push(2, 0, 0);
            var f = [0, 0, 0];
            f[Math.floor(i/2) % 3] = 1;
            fbuffer.push(f[0], f[1], f[2]);
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
            setOpacity: setValue,
            setOpacity: setDelta,
            setOpacity: setOpacity
            
        };
        return result;
    }

    // Exported functionality:
    var contourist = {
        Irregular2D: Irregular2D
    };
debugger;
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