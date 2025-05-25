void Infill::generateZigZagInfill(Polygons& result, const coord_t line_distance, const double& infill_rotation)
{
    const coord_t shift = getShiftOffsetFromInfillOriginAndRotation(infill_rotation);

    PointMatrix rotation_matrix(infill_rotation);
    ZigzagConnectorProcessor zigzag_processor(rotation_matrix, result, use_endpieces, connected_zigzags, skip_some_zags, zag_skip_count);
    generateLinearBasedInfill(outline_offset, result, line_distance, rotation_matrix, zigzag_processor, connected_zigzags, shift);
}


coord_t Infill::getShiftOffsetFromInfillOriginAndRotation(const double& infill_rotation)
{
    if (infill_origin.X != 0 || infill_origin.Y != 0)
    {
        const double rotation_rads = infill_rotation * M_PI / 180;
        return infill_origin.X * std::cos(rotation_rads) - infill_origin.Y * std::sin(rotation_rads);
    }
    return 0;
}


void Infill::generateLinearBasedInfill(const int outline_offset, Polygons& result, const int line_distance, const PointMatrix& rotation_matrix, ZigzagConnectorProcessor& zigzag_connector_processor, const bool connected_zigzags, coord_t extra_shift)
{
    if (line_distance == 0)
    {
        return;
    }
    if (in_outline.size() == 0)
    {
        return;
    }

    coord_t shift = extra_shift + this->shift;

    if (outline_offset != 0 && perimeter_gaps)
    {
        const Polygons gaps_outline = in_outline.offset(outline_offset + infill_line_width / 2 + perimeter_gaps_extra_offset);
        perimeter_gaps->add(in_outline.difference(gaps_outline));
    }

    Polygons outline = in_outline.offset(outline_offset + infill_overlap);

    if (outline.size() == 0)
    {
        return;
    }
    //TODO: Currently we find the outline every time for each rotation.
    //We should compute it only once and rotate that accordingly.
    //We'll also have the guarantee that they have the same size every time.
    //Currently we assume that the above operations are all rotation-invariant,
    //which they aren't if vertices fall on the same coordinate due to rounding.
    crossings_on_line.resize(outline.size()); //One for each polygon.

    outline.applyMatrix(rotation_matrix);

    if (shift < 0)
    {
        shift = line_distance - (-shift) % line_distance;
    }
    else
    {
        shift = shift % line_distance;
    }

    AABB boundary(outline);

    int scanline_min_idx = computeScanSegmentIdx(boundary.min.X - shift, line_distance);
    int line_count = computeScanSegmentIdx(boundary.max.X - shift, line_distance) + 1 - scanline_min_idx;

    std::vector<std::vector<coord_t>> cut_list; // mapping from scanline to all intersections with polygon segments

    for(int scanline_idx = 0; scanline_idx < line_count; scanline_idx++)
    {
        cut_list.push_back(std::vector<coord_t>());
    }

    //When we find crossings, keep track of which crossing belongs to which scanline and to which polygon line segment.
    //Then we can later join two crossings together to form lines and still know what polygon line segments that infill line connected to.
    struct Crossing
    {
        Crossing(Point coordinate, size_t polygon_index, size_t vertex_index): coordinate(coordinate), polygon_index(polygon_index), vertex_index(vertex_index) {};
        Point coordinate;
        size_t polygon_index;
        size_t vertex_index;
        bool operator <(const Crossing& other) const //Crossings will be ordered by their Y coordinate so that they get ordered along the scanline.
        {
            return coordinate.Y < other.coordinate.Y;
        }
    };
    std::vector<std::vector<Crossing>> crossings_per_scanline; //For each scanline, a list of crossings.
    const int min_scanline_index = computeScanSegmentIdx(boundary.min.X - shift, line_distance) + 1;
    const int max_scanline_index = computeScanSegmentIdx(boundary.max.X - shift, line_distance) + 1;
    crossings_per_scanline.resize(max_scanline_index - min_scanline_index);

    for(size_t poly_idx = 0; poly_idx < outline.size(); poly_idx++)
    {
        PolygonRef poly = outline[poly_idx];
        crossings_on_line[poly_idx].resize(poly.size()); //One for each line in this polygon.
        Point p0 = poly.back();
        zigzag_connector_processor.registerVertex(p0); // always adds the first point to ZigzagConnectorProcessorEndPieces::first_zigzag_connector when using a zigzag infill type

        for(size_t point_idx = 0; point_idx < poly.size(); point_idx++)
        {
            Point p1 = poly[point_idx];
            if (p1.X == p0.X)
            {
                zigzag_connector_processor.registerVertex(p1); 
                // TODO: how to make sure it always adds the shortest line? (in order to prevent overlap with the zigzag connectors)
                // note: this is already a problem for normal infill, but hasn't really bothered anyone so far.
                p0 = p1;
                continue; 
            }

            int scanline_idx0;
            int scanline_idx1;
            // this way of handling the indices takes care of the case where a boundary line segment ends exactly on a scanline:
            // in case the next segment moves back from that scanline either 2 or 0 scanline-boundary intersections are created
            // otherwise only 1 will be created, counting as an actual intersection
            int direction = 1;
            if (p0.X < p1.X) 
            {
                scanline_idx0 = computeScanSegmentIdx(p0.X - shift, line_distance) + 1; // + 1 cause we don't cross the scanline of the first scan segment
                scanline_idx1 = computeScanSegmentIdx(p1.X - shift, line_distance); // -1 cause the vertex point is handled in the next segment (or not in the case which looks like >)
            }
            else
            {
                direction = -1;
                scanline_idx0 = computeScanSegmentIdx(p0.X - shift, line_distance); // -1 cause the vertex point is handled in the previous segment (or not in the case which looks like >)
                scanline_idx1 = computeScanSegmentIdx(p1.X - shift, line_distance) + 1; // + 1 cause we don't cross the scanline of the first scan segment
            }

            for(int scanline_idx = scanline_idx0; scanline_idx != scanline_idx1 + direction; scanline_idx += direction)
            {
                int x = scanline_idx * line_distance + shift;
                int y = p1.Y + (p0.Y - p1.Y) * (x - p1.X) / (p0.X - p1.X);
                assert(scanline_idx - scanline_min_idx >= 0 && scanline_idx - scanline_min_idx < int(cut_list.size()) && "reading infill cutlist index out of bounds!");
                cut_list[scanline_idx - scanline_min_idx].push_back(y);
                Point scanline_linesegment_intersection(x, y);
                zigzag_connector_processor.registerScanlineSegmentIntersection(scanline_linesegment_intersection, scanline_idx);
                crossings_per_scanline[scanline_idx - min_scanline_index].emplace_back(scanline_linesegment_intersection, poly_idx, point_idx);
            }
            zigzag_connector_processor.registerVertex(p1);
            p0 = p1;
        }
        zigzag_connector_processor.registerPolyFinished();
    }
    
    //Gather all crossings per scanline and find out which crossings belong together, then store them in crossings_on_line.
    for (int scanline_index = min_scanline_index; scanline_index < max_scanline_index; scanline_index++)
    {
        std::sort(crossings_per_scanline[scanline_index - min_scanline_index].begin(), crossings_per_scanline[scanline_index - min_scanline_index].end()); //Sorts them by Y coordinate.
        for (long crossing_index = 0; crossing_index < static_cast<long>(crossings_per_scanline[scanline_index - min_scanline_index].size()) - 1; crossing_index += 2) //Combine each 2 subsequent crossings together.
        {
            const Crossing& first = crossings_per_scanline[scanline_index - min_scanline_index][crossing_index];
            const Crossing& second = crossings_per_scanline[scanline_index - min_scanline_index][crossing_index + 1];
            //Avoid creating zero length crossing lines
            const Point unrotated_first = rotation_matrix.unapply(first.coordinate);
            const Point unrotated_second = rotation_matrix.unapply(second.coordinate);
            if (unrotated_first == unrotated_second)
            {
                continue;
            }
            InfillLineSegment* new_segment = new InfillLineSegment(unrotated_first, first.vertex_index, first.polygon_index, unrotated_second, second.vertex_index, second.polygon_index);
            //Put the same line segment in the data structure twice: Once for each of the polygon line segment that it crosses.
            crossings_on_line[first.polygon_index][first.vertex_index].push_back(new_segment);
            crossings_on_line[second.polygon_index][second.vertex_index].push_back(new_segment);
        }
    }

    if (cut_list.size() == 0)
    {
        return;
    }
    if (connected_zigzags && cut_list.size() == 1 && cut_list[0].size() <= 2)
    {
        return;  // don't add connection if boundary already contains whole outline!
    }

    addLineInfill(result, rotation_matrix, scanline_min_idx, line_distance, boundary, cut_list, shift);
}