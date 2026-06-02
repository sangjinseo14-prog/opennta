#@File(label="Input file") inputFile
#@Float(label="Quality threshold") qualityThreshold
#@Float(label="Radius") RADIUS
#@Float(label="Linking max distance") LINKING_MAX_DISTANCE
#@Float(label="Gap closing max distance") GAP_CLOSING_MAX_DISTANCE
#@Integer(label="Max frame gap") MAX_FRAME_GAP
#@Integer(label="TRACK_LENGTH_MIN") TRACK_LENGTH_MIN
#@Integer(label="borderMargin") borderMargin

import java.lang.System
import sys
from ij import IJ
from fiji.plugin.trackmate import Model, Settings, TrackMate
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SimpleSparseLAPTrackerFactory
from fiji.plugin.trackmate.features import FeatureFilter
import os
from java.io import File

TAG = u"    #TM "

def log(msg):
    try:
        IJ.log(u"%s%s" % (TAG, unicode(msg)))
    except Exception:
        IJ.log("%s%s" % (TAG, str(msg)))

def main():
    if inputFile is None:
        log(u"[X] inputFile is not specified. (File selection failed)")
        return

    file_path = unicode(inputFile.getAbsolutePath())

    imp = IJ.openImage(file_path)
    if imp is None:
        log(u"[X] Failed to open image: %s" % file_path)
        return

    nframes = imp.getNFrames()
    nstacks = imp.getStackSize()
    if nframes <= 1 and nstacks > 1:
        try:
            imp.setOpenAsHyperStack(True)
            imp.setDimensions(1, 1, nstacks)  # C=1, Z=1, T=nstacks
        except Exception as e:
            log(u"[!] Failed to convert to HyperStack: %s" % unicode(e))

    log(u"Opening: %s" % unicode(imp.getTitle()))
    imp.show()

    model = Model()
    settings = Settings(imp)
    settings.detectorFactory = LogDetectorFactory()
    settings.trackerFactory = SimpleSparseLAPTrackerFactory()
    settings.addAllAnalyzers()

    settings.detectorSettings = {
        'RADIUS': float(RADIUS),
        'THRESHOLD': float(qualityThreshold),
        'DO_MEDIAN_FILTERING': False,
        'DO_SUBPIXEL_LOCALIZATION': True,
        'TARGET_CHANNEL': 1
    }

    det_keys = ['RADIUS', 'THRESHOLD', 'DO_SUBPIXEL_LOCALIZATION', 'DO_MEDIAN_FILTERING', 'TARGET_CHANNEL']
    log("Detector: " + str([settings.detectorSettings.get(k) for k in det_keys]))

    settings.trackerSettings = {
        'LINKING_MAX_DISTANCE': float(LINKING_MAX_DISTANCE),
        'GAP_CLOSING_MAX_DISTANCE': float(GAP_CLOSING_MAX_DISTANCE),
        'MAX_FRAME_GAP': int(MAX_FRAME_GAP),
        'ALLOW_GAP_CLOSING': True,
        'ALLOW_TRACK_SPLITTING': False,
        'SPLITTING_MAX_DISTANCE': 6.0,
        'ALLOW_TRACK_MERGING': False,
        'MERGING_MAX_DISTANCE': 6.0,
        'BLOCKING_VALUE': float('inf'),
        'CUTOFF_PERCENTILE': 0.9,
        'ALTERNATIVE_LINKING_COST_FACTOR': 1.05
    }

    w, h = imp.getWidth(), imp.getHeight()
    min_x = int(0 + borderMargin)
    max_x = int(w - borderMargin)
    min_y = int(0 + borderMargin)
    max_y = int(h - borderMargin)

    tra_keys = ['LINKING_MAX_DISTANCE','GAP_CLOSING_MAX_DISTANCE','MAX_FRAME_GAP',
            'ALLOW_GAP_CLOSING','ALLOW_TRACK_SPLITTING','ALLOW_TRACK_MERGING']
    log("Tracker: " + str([settings.trackerSettings.get(k) for k in tra_keys]))

    settings.addTrackFilter(FeatureFilter("TRACK_DURATION", TRACK_LENGTH_MIN, True))
    settings.addTrackFilter(FeatureFilter("X_LOCATION", min_x, True))
    settings.addTrackFilter(FeatureFilter("X_LOCATION", max_x, False))
    settings.addTrackFilter(FeatureFilter("Y_LOCATION", min_y, True))
    settings.addTrackFilter(FeatureFilter("Y_LOCATION", max_y, False))

    def _fmt_num(v):
        try:
            v = float(v)
            return ("%d" % int(round(v))) if abs(v - round(v)) < 1e-6 else ("%.3f" % v)
        except (ValueError, TypeError):
            return str(v)

    def _log_one_line_track_filters(flist):
        if flist is None or (hasattr(flist, "isEmpty") and flist.isEmpty()):
            log("Tracker Filter: (none)")
            return

        parts = []
        for ff in list(flist):
            try:
                feat = ff.getFeature()
            except Exception:
                feat = getattr(ff, "feature", "UNKNOWN")

            val = None
            for attr in ("getThreshold", "getValue", "threshold", "value"):
                try:
                    v = getattr(ff, attr)
                    val = float(v() if callable(v) else v)
                    break
                except (AttributeError, TypeError, ValueError):
                    pass

            try:
                above = bool(ff.isAbove())
            except Exception:
                above = bool(getattr(ff, "isAbove", True))

            op = ">=" if above else "<="
            parts.append((feat, op, val))

        order = {"X_LOCATION": 0, "Y_LOCATION": 1, "TRACK_DURATION": 2}
        parts.sort(key=lambda t: (order.get(t[0], 99), t[0], t[1]))

        msg = ";  ".join("%s %s %s" % (f, o, _fmt_num(v) if v is not None else "NaN")
                         for (f, o, v) in parts)
        log(u"Tracker Filter: " + msg)

    _log_one_line_track_filters(settings.getTrackFilters())

    trackmate = TrackMate(model, settings)
    if not trackmate.checkInput():
        log("[X] Input check failed: " + unicode(trackmate.getErrorMessage()))
        return
    if not trackmate.process():
        log("[X] TrackMate processing failed: " + unicode(trackmate.getErrorMessage()))
        return

    # Only spots that survive track filtering — TrackMate keeps the rejected
    # ones in the model, so dump via track_model.trackSpots, not getSpots().
    base, _ = os.path.splitext(file_path)
    output_csv = base + "_spots.csv"
    fname = os.path.basename(output_csv)
    track_model = model.getTrackModel()
    with open(output_csv, 'w') as f:
        for track_id in track_model.trackIDs(True):
            for spot in track_model.trackSpots(track_id):
                x = spot.getFeature('POSITION_X')
                y = spot.getFeature('POSITION_Y')
                frame = int(spot.getFeature('FRAME'))
                q = spot.getFeature('QUALITY')
                f.write('%d,%.3f,%.3f,%d,%.6f\n' % (track_id, x, y, frame, q))

    total_spots = model.getSpots().getNSpots(True)
    total_tracks = model.getTrackModel().nTracks(False)
    filtered_tracks = model.getTrackModel().nTracks(True)

    log("Results saved: %s" % unicode(fname))
    log("Total spots: %d, Total tracks: %d, Filtered tracks: %d" %
        (total_spots, total_tracks, filtered_tracks))

    imp.close()
    java.lang.System.gc()

main()
