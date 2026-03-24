let
  pkgs = import <nixpkgs> { config.allowUnfree = true; };
  python = pkgs.python312;

  lib-path = with pkgs; lib.makeLibraryPath [
    libffi
    openssl
    stdenv.cc.cc
    zlib
    glib
  ];

in pkgs.mkShell {
  packages = [
    (python.withPackages (ps: with ps; [
      numpy
      pandas
      xgboost
      scikit-learn
      matplotlib
      seaborn
      notebook
      ipykernel
      jupyterlab
      statsmodels
    ]))

    pkgs.git
    pkgs.zlib
    pkgs.libffi
    pkgs.openssl
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${lib-path}:$LD_LIBRARY_PATH"

    export PYTHONPATH=$PWD

    python -m ipykernel install --user --name=nix-ml --display-name "Python (nix-ml)"

  '';
}
