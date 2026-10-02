function drawScene(c) {
  if (index === 0) editorialCover(c);
  else if (index === 18) editorialClosing(c);
  else if (index === 11) jointMotion(c);
  else if (index === 16) quantMotion(c);
  else if (index === 17) limitsMotion(c);
  else if (index === 3) relatedMotion(c);
  else if (index >= 4 && index <= 8) methodMotion(c);
  else if (index === 9 || index === 10) flowMotion(c);
  else if (charts[index]) benchmarkMotion(c);
  else if (index === 2) humanRobotScene(c);
}
