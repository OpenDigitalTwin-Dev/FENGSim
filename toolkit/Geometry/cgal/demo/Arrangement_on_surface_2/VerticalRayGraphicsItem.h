// Copyright (c) 2012, 2020 Tel-Aviv University (Israel).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2.2/Arrangement_on_surface_2/demo/Arrangement_on_surface_2/VerticalRayGraphicsItem.h $
// $Id: VerticalRayGraphicsItem.h a30658a 2020-09-21T09:09:48+02:00 Ahmed Essam
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s): Alex Tsui <alextsui05@gmail.com>
//            Ahmed Essam <theartful.ae@gmail.com>

#ifndef VERTICAL_RAY_GRAPHICS_ITEM_H
#define VERTICAL_RAY_GRAPHICS_ITEM_H

#include <CGAL/Qt/GraphicsItem.h>
#include <QPen>
#include <QColor>

/*
 * Represents a vertical ray in the scene. The ray doesn't necessarily extend
 * to infinity, but it's called a ray because it has an arrowhead at its target
 * end.
 *
 * If it is designated that the ray extends to infinity, we'll clip the ray to
 * the boundary of the visible viewport.
 */
class VerticalRayGraphicsItem : public CGAL::Qt::GraphicsItem
{
public:
  VerticalRayGraphicsItem( );

  virtual void paint( QPainter* painter,
                      const QStyleOptionGraphicsItem* option,
                      QWidget* widget );
  virtual QRectF boundingRect( ) const;

  const QPointF& source( ) const;
  void setSource( const QPointF& src );
  double targetY( ) const;
  void setTargetY( double y );
  bool isInfinite( ) const;
  void setIsInfinite( bool b );

  const QColor& color( ) const;
  void setColor( const QColor& color );
  int width( ) const;
  void setWidth( int width );

  void reset( );

public Q_SLOTS:
  virtual void modelChanged( );

protected:
  QRectF viewportRect( ) const;
  void drawArrowhead( QPainter* painter, double targetY, bool isShootingUp );

  QPointF m_source;
  double m_targetY;
  bool m_isInfinite;
  QColor m_color;
  int m_width;
}; // class VerticalRayGraphicsItem

#endif // VERTICAL_RAY_GRAPHICS_ITEM_H
