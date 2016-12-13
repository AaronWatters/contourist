
THREE.morph_sequence = function(morph_data, scene, duration, material) {
    var morph_descriptions = morph_data["morph_descriptions"];
    var global_max = morph_data["max_value"];
    var global_min = morph_data["min_value"];
    var value_change = global_max - global_min;
    var index = 0;
    var index_started = 0.0;
    var mesh = null;
    var clock = new THREE.Clock();
    clock.start();

    var start_segment = function() {
        var morph_info = morph_descriptions[index];
        var max_value = morph_info["max_value"];
        var min_value = morph_info["min_value"];
        var start_positions = morph_info["start_positions"];
        var end_positions = morph_info["end_positions"];
        var indices = morph_info["triangles"];
        var geometry = new THREE.Geometry();
        var geomv = geometry.vertices;
        var morphv = [];
        for (var i=0; i<start_positions.length; i++) {
            var v = start_positions[i];
            geomv.push(new THREE.Vector3(v[0], v[1], v[2]));
            var mv = end_positions[i];
            morphv.push(new THREE.Vector3(mv[0], mv[1], mv[2]));
        }
        geometry.morphTargets.push( { name: "target0", vertices: morphv } );
        var geomf = geometry.faces;
        for (var i=0; i<indices.length; i++) {
            var f = indices[i];
            var face = new THREE.Face3(f[0], f[1], f[2]);
            geomf.push(face);
        }
        geometry.computeFaceNormals();
        geometry.computeVertexNormals();
        geometry.computeMorphNormals();
        // patch the normals
        //debugger;
        var gfaces = geometry.faces;
        var morphNormals = geometry.morphNormals[0];
        var mfaceNormals = morphNormals.faceNormals;
        var mvertexNormals = morphNormals.vertexNormals;
        var patch_if_zero = function (normal, source) {
            if (normal.x == 0 && normal.y == 0 && normal.z == 0) {
                normal.copy(source);
            }
        };
        for (var i = 0; i < gfaces.length; i++) {
            var fnormal = gfaces[i].normal;
            var fvertexNormalsi = gfaces[i].vertexNormals;
            var mvertexNormalsi = mvertexNormals[i];
            patch_if_zero(fnormal, mfaceNormals[i]);
            var mvertexNormalsi = mvertexNormals[i];
            patch_if_zero(fvertexNormalsi[0], mvertexNormalsi.a);
            patch_if_zero(fvertexNormalsi[1], mvertexNormalsi.b);
            patch_if_zero(fvertexNormalsi[2], mvertexNormalsi.c);
        }
        var rotation = null;
        if (mesh != null) {
            rotation = mesh.rotation;
            scene.remove(mesh);
        }
        mesh = new THREE.Mesh( geometry, material );
        if (rotation != null) {
            mesh.rotation.copy(rotation);
        }
        scene.add(mesh);
        morph_info["morph_duration"] = (duration * (max_value - min_value)) / value_change;
        morph_info["started_at"] = clock.getElapsedTime();
        morph_info["end_at"] = morph_info["started_at"] + morph_info["morph_duration"]
        return mesh;
    }

    mesh = start_segment();

    var tick = function() {
        var morph_info = morph_descriptions[index];
        var end_at = morph_info["end_at"];
        var now = clock.getElapsedTime();
        if (now > end_at) {
            index = (index + 1) % morph_descriptions.length;
            console.log("starting segment " + index);
            start_segment();
        } else {
            var started_at = morph_info["started_at"];
            var morph_duration = morph_info["morph_duration"];
            var influence = (now - started_at)/morph_duration;
            mesh.morphTargetInfluences[ 0 ] = influence;
        }
    }

    var get_mesh = function() {
        return mesh;
    }

    return {
        tick: tick,
        get_mesh: get_mesh
    }
}