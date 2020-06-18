from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        :param rects: list of bounding box rectangles, tuple of form (startX, startY, endX, endY)
        :return:
        """
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX+endX) / 2.0)
            cY = int((startY+endY) / 2.0)
            inputCentroids[i] = (cX, cY)

            if len(self.objects == 0):
                for i in range(0, len(inputCentroids)):
                    self.register(inputCentroids[i])
            else:
                objectIDs = list(self.objects.keys())
                objectCentroids = list(self.objects.values())

                D = dist.cdist(np.array(objectCentroids), inputCentroids)
                rows = D.min(axis=1).argsort()

                cols = D.argmin(axis=1)[rows]
                usedRows = set()
                usedCols = set()
                # loop over the combination of the (row, column) index
                # tuples
                for (row, col) in zip(rows, cols):
                    # if we have already examined either the row or
                    # column value before, ignore it
                    # val
                    if row in usedRows or col in usedCols:
                        continue
                    # otherwise, grab the object ID for the current row,
                    # set its new centroid, and reset the disappeared
                    # counter
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    # indicate that we have examined each of the row and
                    # column indexes, respectively
                    usedRows.add(row)
                    usedCols.add(col)
