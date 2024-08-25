[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

<p align="left">
    <img src="logo.png" alt="Logo" align="center">
</p>

Sprite and Text based Graphical Interface. A glorified sprite and text renderer to cook up sprite based game uis. Pixelperfect cursor detection included, transparency supported.

Basically, STGI allows you to define ui areas and put a texture and text on them. It also detects if the cursor hovers over non-transparent parts of the texture. This allows pixel perfect mouse hit detection for sprite based user interfaces.

STGI is fairly well optimized, with a main focus on static sprites. It also supports animated sprites (not yet) and rendering text (not yet).

## Demo
![hello_stgi example](./showcase.png)

## Features
- **Flexible and simple to use (could be utilized for more than just ui)**
- **Extensively  hardware accelerated**
- **Pixelperfect cursor hit detection**
- **Supports transparency**
- **Well optimized**
- **No native-only wgpu features used (good for wasm)**
- **Windowing library independend**
- **Integrated text layout and rendering (NOT YET IMPLEMENTED)**
- **Supports animated sprites (NOT YET IMPLEMENTED)**

## Examples
To run the example in the repository, run the following command:
```bash
cargo run --example hello_stgi --release
```

## FAQ

#### Why the assault rifle?

The assault rifle in the logo is a Steyr AUG, also known as StG-77 (Sturmgewehr-77). I used it as a logo for three reasons:
- STGI and StG is a cool word play
- It's the rifle I had while in the military
- It fits the whole pixel perfect hit detection and game ui theme

#### Isn't this just a glorified sprite (and text) renderer?

Well, apart from the cursor hit detection, yes. It's a simple to use sprite and glyph renderer designed to be used for simple, sprite based uis. Think Dark Souls; Dark Souls used a sprite based ui and the game is goated.