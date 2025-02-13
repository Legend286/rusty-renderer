struct VertexInput
{
    @location(0) position: vec4<f32>,
    @location(1) color : vec4<f32>,
};

struct VertexOutput
{
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct PixelOutput
{
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput
{
    var output: VertexOutput;
    output.position = vec4<f32>(input.position.xyz, 1.0); // Pass position to pipeline
    output.color = input.color; // Pass color to fragment shader
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> PixelOutput
{
    var output: PixelOutput;
    output.color = input.color;
    return output; // Output fragment color
}
