"""Shared test helpers."""
import importlib
import pkgutil


def patch_service_areas(monkeypatch, name, value):
    """Replace ``name`` wherever the service-area code looks it up.

    The service-area code is split over ``wwtp.service_areas.*`` (and the CLI in
    ``wwtp.create_voronoi``); a function finds a collaborator in its own module, so
    a stub must replace it in every module that holds the name.
    """
    import wwtp.create_voronoi as cli
    import wwtp.service_areas as pkg

    modules = [cli, pkg] + [importlib.import_module(f"{pkg.__name__}.{m.name}") for m in pkgutil.iter_modules(pkg.__path__)]
    hits = [mod for mod in modules if name in vars(mod)]
    if not hits:
        raise AttributeError(f"no service-area module defines {name!r}")
    for mod in hits:
        monkeypatch.setattr(mod, name, value)
