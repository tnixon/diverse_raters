def test_package_imports():
    import dre
    import dre.settings

    assert dre.__version__
    assert callable(dre.settings.configure_mlflow)
