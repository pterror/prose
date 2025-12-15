{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };
  outputs =
    { self, nixpkgs }:
    let
      forAllSystems =
        with nixpkgs.lib;
        f: foldAttrs mergeAttrs { } (map (s: { ${s} = f s; }) systems.flakeExposed);
    in
    {
      devShell = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true; # Required for CUDA packages
          };
        in
        pkgs.mkShell rec {
          packages = with pkgs; [
            psmisc
            ripgrep
            stdenv.cc.cc
            python313
            uv
            ruff
            zlib

            # CUDA support for GPU training
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
          ];

          LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath (
              packages
              ++ [
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ]
            )
            + ":$LD_LIBRARY_PATH";
          shellHook = ''
            export PATH=".venv/bin:$PATH"

            # Add NVIDIA driver paths for GPU detection
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          '';
        }
      );
    };
}
