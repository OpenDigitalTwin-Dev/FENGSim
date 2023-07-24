// Copyright (c) 2012, 2020 Tel-Aviv University (Israel).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Arrangement_on_surface_2/demo/Arrangement_on_surface_2/ArrangementDemoGraphicsView.h $
// $Id: ArrangementDemoGraphicsView.h a30658a 2020-09-21T09:09:48+02:00 Ahmed Essam
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s): Alex Tsui <alextsui05@gmail.com>
//            Ahmed Essam <theartful.ae@gmail.com>

#ifndef ARRANGEMENT_DEMO_GRAPHICS_VIEW_H
#define ARRANGEMENT_DEMO_GRAPHICS_VIEW_H

#include <QGraphicsView>
#include <QColor>

class QPaintEvent;

class ArrangementDemoGraphicsView : public QGraphicsView
{
public:
  ArrangementDemoGraphicsView( QWidget* parent = 0 );

  void setBackgroundColor( QColor color );
  QColor getBackgroundColor( ) const;
  void resetTransform();

protected:
  void paintEvent(QPaintEvent* event) override;

  qreal maxScale;
  qreal minScale;
};

#endif // ARRANGEMENT_DEMO_GRAPHICS_VIEW_H
