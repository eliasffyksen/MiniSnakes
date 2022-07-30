{
  description = "RL env bench";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-22.05-darwin";
    flake-utils.url = "github:numtide/flake-utils";
    devshell.url = "github:numtide/devshell";
  };

  outputs = { self, nixpkgs, flake-utils, devshell }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = import nixpkgs {
          inherit system;
          overlays = [ devshell.overlay ];
        }; in {


          devShell = pkgs.mkShell {

            nativeBuildInputs =
              with pkgs;
              with python310Packages;
            [
              ninja
              clang_14
              cmake

              python310
              pybind11
              pytorch
              tqdm
              matplotlib
            ];
          };
        }
      );
}
