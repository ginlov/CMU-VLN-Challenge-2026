"""Namespace stub — deliberately not the development repo's `__init__.py`.

Upstream this module re-exports `PerceptionResponder`, which imports the
sidecar client (httpx, pillow, pycocotools) and the shared scene
representation. The drive loop needs two leaves out of this package and
nothing else:

    from xiao_hei_vln.perception.geometry   import sensor_to_camera_transform
    from xiao_hei_vln.perception.size_prior import size_for

Re-exporting the responder here would make both of those imports drag the
whole sidecar stack into the ai_module image to satisfy a call that never
happens. So this file stays empty on purpose, and `sync_ai_module.sh` does
not overwrite it.
"""
