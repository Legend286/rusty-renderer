struct VertexOutput
{
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> VertexOutput
{
    var output: VertexOutput;
    output.position = vec4<f32>(position, 1.0); // Pass position to pipeline
    output.color = color; // Pass color to fragment shader
    return output;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32>
{
    return vec4<f32>(color, 1.0); // Output fragment color
}
