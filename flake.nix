{
  inputs = {
    # Use git+ssh protocol because it's a private repository
    # See https://discourse.nixos.org/t/nix-flakes-and-private-repositories/12014
    ml-ops.url = "git+ssh://git@github.com/Preemo-Inc/ml-ops.git";
    ml-ops.inputs.systems.url = "github:nix-systems/default";
  };
  outputs = inputs @ { ml-ops, ... }:
    ml-ops.lib.mkFlake { inherit inputs; } {
      imports = [
        ml-ops.flakeModules.devcontainer
        ml-ops.flakeModules.nixIde
        ml-ops.flakeModules.nixLd
        ml-ops.flakeModules.pythonVscode
        ml-ops.flakeModules.devenvPythonWithLibstdcxx
      ];
      perSystem = { pkgs, config, lib, system, ... }: {
        ml-ops.devcontainer = {
          nixago.requests.".vscode/settings.json".data = {
            "python.defaultInterpreterPath" = "./.venv/bin/python";
            "python.analysis.typeCheckingMode" = "strict";
          };

          devenvShellModule = {
            languages = {
              python = {
                enable = true;
                poetry = {
                  enable = true;
                };
              };
            };
          };
        };

      };
    };
}
