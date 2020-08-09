for (objectID, centroid) in objects.items():
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
# Draw label
label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
              cv2.FILLED)  # Draw white box to put label text in
cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
            2)  # Draw label text