#@File(label="Input file") inputFile
#@Float(label="Radius") RADIUS

import java.lang.System
import sys
from ij import IJ
from fiji.plugin.trackmate import Model, Settings, TrackMate
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SimpleSparseLAPTrackerFactory

import os

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

    log(u"Opening: %s" % unicode(imp.getTitle()))

    model = Model()
    settings = Settings(imp)

    settings.detectorFactory = LogDetectorFactory()
    settings.trackerFactory = SimpleSparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()

    settings.detectorSettings = {
        'RADIUS': float(RADIUS),
        'THRESHOLD': -500.0,
        'DO_MEDIAN_FILTERING': False,
        'DO_SUBPIXEL_LOCALIZATION': True,
        'TARGET_CHANNEL': 1
    }

    trackmate = TrackMate(model, settings)

    if not trackmate.checkInput():
        log("[X] Input check failed: " + unicode(trackmate.getErrorMessage()))
        return

    if not trackmate.execDetection():
        log("[X] Detection failed: " + unicode(trackmate.getErrorMessage()))
        return

    log("Detection completed.")

    base, _ = os.path.splitext(file_path)
    output_csv = base + "_quality.csv"
    fname = os.path.basename(output_csv)
    spots = model.getSpots()
    it0 = spots.iterator(0, False)

    count = 0
    with open(output_csv, 'w') as f:
        f.write("X,Y,QUALITY\n")
        while it0.hasNext():
            sp = it0.next()
            x = sp.getFeature('POSITION_X')
            y = sp.getFeature('POSITION_Y')
            q = sp.getFeature('QUALITY')
            f.write("{},{},{}\n".format(x, y, q))
            count += 1

    log(u"Quality saved: %d  ->  %s" % (int(count), unicode(fname)))

    imp.close()
    java.lang.System.gc()

main()
