{
  description = "Entorno para la actividad de analisis multivariado (Estadistica 2): pandas, scikit-learn y matplotlib";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # Shell de desarrollo: nix develop
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            pandas
            openpyxl
            scikit-learn
            matplotlib
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            env = {
              # Matplotlib sin ventana grafica: guarda los PNG y no intenta abrir una GUI
              MPLBACKEND = "Agg";
              # Evita que Python escriba cache .pyc fuera de control de versiones
              PYTHONDONTWRITEBYTECODE = "1";
            };
          };
        });

      # Ejecucion directa sin entrar al shell: nix run
      apps = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            pandas
            openpyxl
            scikit-learn
            matplotlib
          ]);
          runActividad = pkgs.writeShellScript "run-actividad" ''
            cd ${self}
            export MPLBACKEND=Agg
            exec ${pythonEnv}/bin/python main.py "$@"
          '';
        in
        {
          default = {
            type = "app";
            program = toString runActividad;
          };
        });
    };
}

