/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#pragma once

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/impl/instantiate.hpp>
#include "voxel_grid_indices.h"


struct cloud_point_index_idx_ul
{
    int64_t idx;
    unsigned int cloud_point_index;

    cloud_point_index_idx_ul (unsigned int idx_, unsigned int
    cloud_point_index_) : idx (idx_), cloud_point_index (cloud_point_index_) {}
    bool operator < (const cloud_point_index_idx_ul &p) const { return (idx <
                                                                        p.idx); }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void pcl::VoxelGridIndices<PointT>::applyFilter(PointCloud &output) {
    /// Temporary fast solution
    PointIndices::Ptr indices(new PointIndices);
    this->filter(indices->indices);

    pcl::copyPointCloud(*input_, indices->indices, output);

    /*
    // Has the input dataset been set already?
    if (!input_)
    {
      PCL_WARN ("[pcl::%s::applyFilter] No input dataset given!\n", getClassName
    ().c_str ());
      output.width = output.height = 0;
      output.points.clear ();
      return;
    }

    // Copy the header (and thus the frame_id) + allocate enough space for points
    output.height       = 1;                    // downsampling breaks the
    organized structure
    output.is_dense     = true;                 // we filter out invalid points

    Eigen::Vector4f min_p, max_p;
    // Get the minimum and maximum dimensions
    if (!filter_field_name_.empty ()) // If we don't want to process the entire
    cloud...
      getMinMax3D<PointT> (input_, *indices_, filter_field_name_,
    static_cast<float> (filter_limit_min_), static_cast<float>
    (filter_limit_max_), min_p, max_p, filter_limit_negative_);
    else
      getMinMax3D<PointT> (*input_, *indices_, min_p, max_p);

    // Check that the leaf size is not too small, given the size of the data
    int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) *
    inverse_leaf_size_[0])+1;
    int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) *
    inverse_leaf_size_[1])+1;
    int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) *
    inverse_leaf_size_[2])+1;

    if ((dx*dy*dz) > (std::numeric_limits<int64_t>::max()))
    {
      PCL_WARN("[pcl::%s::applyFilter] Leaf size is too small for the input
    dataset. Integer indices would overflow.", getClassName().c_str());
      output = *input_;
      return;
    }

    // Compute the minimum and maximum bounding box values
    min_b_[0] = static_cast<int64_t> (floor (min_p[0] * inverse_leaf_size_[0]));
    max_b_[0] = static_cast<int64_t> (floor (max_p[0] * inverse_leaf_size_[0]));
    min_b_[1] = static_cast<int64_t> (floor (min_p[1] * inverse_leaf_size_[1]));
    max_b_[1] = static_cast<int64_t> (floor (max_p[1] * inverse_leaf_size_[1]));
    min_b_[2] = static_cast<int64_t> (floor (min_p[2] * inverse_leaf_size_[2]));
    max_b_[2] = static_cast<int64_t> (floor (max_p[2] * inverse_leaf_size_[2]));

    // Compute the number of divisions needed along all axis
    div_b_ = max_b_ - min_b_ + Eigen::Vector4l::Ones ();
    div_b_[3] = 0;

    // Set up the division multiplier
    divb_mul_ = Eigen::Vector4l (1, div_b_[0], div_b_[0] * div_b_[1], 0);

    // Storage for mapping leaf and pointcloud indexes
    std::vector<cloud_point_index_idx_ul> index_vector;
    index_vector.reserve (indices_->size ());

    // If we don't want to process the entire cloud, but rather filter points far
    away from the viewpoint first...
    if (!filter_field_name_.empty ())
    {
      // Get the distance field index
      std::vector<pcl::PCLPointField> fields;
      int distance_idx = pcl::getFieldIndex (*input_, filter_field_name_, fields);
      if (distance_idx == -1)
        PCL_WARN ("[pcl::%s::applyFilter] Invalid filter field name. Index is
    %d.\n", getClassName ().c_str (), distance_idx);

      // First pass: go over all points and insert them into the index_vector
    vector
      // with calculated idx. Points with the same idx value will contribute to
    the
      // same point of resulting CloudPoint
      for (std::vector<int>::const_iterator it = indices_->begin (); it !=
    indices_->end (); ++it)
      {
        if (!input_->is_dense)
          // Check if the point is invalid
          if (!pcl_isfinite (input_->points[*it].x) ||
              !pcl_isfinite (input_->points[*it].y) ||
              !pcl_isfinite (input_->points[*it].z))
            continue;

        // Get the distance value
        const uint8_t* pt_data = reinterpret_cast<const uint8_t*>
    (&input_->points[*it]);
        float distance_value = 0;
        memcpy (&distance_value, pt_data + fields[distance_idx].offset, sizeof
    (float));

        if (filter_limit_negative_)
        {
          // Use a threshold for cutting out points which inside the interval
          if ((distance_value < filter_limit_max_) && (distance_value >
    filter_limit_min_))
            continue;
        }
        else
        {
          // Use a threshold for cutting out points which are too close/far away
          if ((distance_value > filter_limit_max_) || (distance_value <
    filter_limit_min_))
            continue;
        }

        int64_t ijk0 = static_cast<int64_t> (floor (input_->points[*it].x *
    inverse_leaf_size_[0]) - static_cast<float> (min_b_[0]));
        int64_t ijk1 = static_cast<int64_t> (floor (input_->points[*it].y *
    inverse_leaf_size_[1]) - static_cast<float> (min_b_[1]));
        int64_t ijk2 = static_cast<int64_t> (floor (input_->points[*it].z *
    inverse_leaf_size_[2]) - static_cast<float> (min_b_[2]));

        // Compute the centroid leaf index
        int64_t idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 *
    divb_mul_[2];
        index_vector.push_back (cloud_point_index_idx_ul (static_cast<int64_t>
    (idx), *it));
      }
    }
    // No distance filtering, process all data
    else
    {
      // First pass: go over all points and insert them into the index_vector
    vector
      // with calculated idx. Points with the same idx value will contribute to
    the
      // same point of resulting CloudPoint
      for (std::vector<int>::const_iterator it = indices_->begin (); it !=
    indices_->end (); ++it)
      {
        if (!input_->is_dense)
          // Check if the point is invalid
          if (!pcl_isfinite (input_->points[*it].x) ||
              !pcl_isfinite (input_->points[*it].y) ||
              !pcl_isfinite (input_->points[*it].z))
            continue;

        int64_t ijk0 = static_cast<int64_t> (floor (input_->points[*it].x *
    inverse_leaf_size_[0]) - static_cast<float> (min_b_[0]));
        int64_t ijk1 = static_cast<int64_t> (floor (input_->points[*it].y *
    inverse_leaf_size_[1]) - static_cast<float> (min_b_[1]));
        int64_t ijk2 = static_cast<int64_t> (floor (input_->points[*it].z *
    inverse_leaf_size_[2]) - static_cast<float> (min_b_[2]));

        // Compute the centroid leaf index
        int64_t idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 *
    divb_mul_[2];
        index_vector.push_back (cloud_point_index_idx_ul (static_cast<int64_t>
    (idx), *it));
      }
    }

    // Second pass: sort the index_vector vector using value representing target
    cell as index
    // in effect all points belonging to the same output cell will be next to each
    other
    std::sort (index_vector.begin (), index_vector.end (),
    std::less<cloud_point_index_idx_ul>());

    // Third pass: count output cells
    // we need to skip all the same, adjacent idx values
    int64_t total = 0;
    int64_t index = 0;
    // first_and_last_indices_vector[i] represents the index in index_vector of
    the first point in
    // index_vector belonging to the voxel which corresponds to the i-th output
    point,
    // and of the first point not belonging to.
    std::vector<std::pair<int64_t, int64_t> > first_and_last_indices_vector;
    // Worst case size
    first_and_last_indices_vector.reserve (index_vector.size ());
    while (index < index_vector.size ())
    {
      int64_t i = index + 1;
      while (i < index_vector.size () && index_vector[i].idx ==
    index_vector[index].idx)
        ++i;
      if (i - index >= min_points_per_voxel_)
      {
        ++total;
        first_and_last_indices_vector.push_back (std::pair<int64_t, int64_t>
    (index, i));
      }
      index = i;
    }

    // Fourth pass: compute centroids, insert them into their final position
    output.points.resize (total);
    if (save_leaf_layout_)
    {
      try
      {
        // Resizing won't reset old elements to -1.  If leaf_layout_ has been used
    previously, it needs to be re-initialized to -1
        int64_t new_layout_size = div_b_[0]*div_b_[1]*div_b_[2];
        //This is the number of elements that need to be re-initialized to -1
        int64_t reinit_size = std::min (static_cast<int64_t> (new_layout_size),
    static_cast<int64_t> (leaf_layout_.size()));
        for (int64_t i = 0; i < reinit_size; i++)
        {
          leaf_layout_[i] = -1;
        }
        leaf_layout_.resize (new_layout_size, -1);
      }
      catch (std::bad_alloc&)
      {
        throw PCLException("VoxelGridIndices bin size is too low; impossible to
    allocate memory for layout",
          "voxel_grid.hpp", "applyFilter");
      }
      catch (std::length_error&)
      {
        throw PCLException("VoxelGridIndices bin size is too low; impossible to
    allocate memory for layout",
          "voxel_grid.hpp", "applyFilter");
      }
    }

    index = 0;
    for (size_t cp = 0; cp < first_and_last_indices_vector.size (); ++cp)
    {
      // calculate centroid - sum values from all input points, that have the same
    idx value in index_vector array
      int64_t first_index = first_and_last_indices_vector[cp].first;
      int64_t last_index = first_and_last_indices_vector[cp].second;

      // index is centroid final position in resulting PointCloud
      if (save_leaf_layout_)
        leaf_layout_[index_vector[first_index].idx] = index;

      //Limit downsampling to coords
      if (!downsample_all_data_)
      {
        Eigen::Vector4f centroid (Eigen::Vector4f::Zero ());

        for (int64_t li = first_index; li < last_index; ++li)
          centroid +=
    input_->points[index_vector[li].cloud_point_index].getVector4fMap ();

        centroid /= static_cast<float> (last_index - first_index);
        output.points[index].getVector4fMap () = centroid;
      }
      else
      {
        CentroidPoint<PointT> centroid;

        // fill in the accumulator with leaf points
        for (int64_t li = first_index; li < last_index; ++li)
          centroid.add (input_->points[index_vector[li].cloud_point_index]);

        centroid.get (output.points[index]);
      }

      ++index;
    }
    output.width = static_cast<uint32_t> (output.points.size ());
     */
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// template <typename PointT> void
// pcl::PassThrough<PointT>::applyFilterIndices (std::vector<int> &indices)
template <typename PointT>
void pcl::VoxelGridIndices<PointT>::applyFilterIndices(
        std::vector<int> &indices) {
    indices.clear();
    removed_indices_->clear();
    // The arrays to be used
    indices.reserve(indices_->size());
    removed_indices_->reserve(indices_->size());

    // Has the input dataset been set already?
    if (!input_) {
        PCL_WARN("[pcl::%s::applyFilter] No input dataset given!\n",
                 getClassName().c_str());
        indices.shrink_to_fit();
        removed_indices_->shrink_to_fit();
        return;
    }

    Eigen::Vector4f min_p, max_p;
    // Get the minimum and maximum dimensions
    if (!filter_field_name_
            .empty())  // If we don't want to process the entire cloud...
        getMinMax3D<PointT>(input_, *indices_, filter_field_name_,
                            static_cast<float>(filter_limit_min_),
                            static_cast<float>(filter_limit_max_), min_p, max_p,
                            filter_limit_negative_);
    else
        getMinMax3D<PointT>(*input_, *indices_, min_p, max_p);

    // Check that the leaf size is not too small, given the size of the data
    int64_t dx =
            static_cast<int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0]) + 1;
    int64_t dy =
            static_cast<int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1]) + 1;
    int64_t dz =
            static_cast<int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2]) + 1;

    if ((dx * dy * dz) > (std::numeric_limits<int64_t>::max())) {
        PCL_WARN(
                "[pcl::%s::applyFilter] Leaf size is too small for the input dataset. "
                        "Integer indices would overflow.",
                getClassName().c_str());
        indices.shrink_to_fit();
        removed_indices_->shrink_to_fit();
        return;
    }

    // Allign bounding boxes between different executions of voxel grid with same
    // leaf_size
    //  Eigen::Vector3f min_grid = min_p.head<3>() / leaf_size_.head<3>();
    //  Eigen::Vector3f max_grid = max_p.head<3>() / leaf_size_.head<3>();
    Eigen::Vector3f min_grid = {min_p[0] / leaf_size_[0],
                                min_p[1] / leaf_size_[1],
                                min_p[2] / leaf_size_[2]};
    Eigen::Vector3f max_grid = {max_p[0] / leaf_size_[0],
                                max_p[1] / leaf_size_[1],
                                max_p[2] / leaf_size_[2]};
    for (int i = 0; i < 3; ++i) {
        min_grid[i] =
                min_grid[i] > 0 ? std::ceil(min_grid[i]) : std::floor(min_grid[i]);
        max_grid[i] =
                max_grid[i] > 0 ? std::ceil(max_grid[i]) : std::floor(max_grid[i]);
    }
    //  min_p.head<3>() = min_grid * leaf_size_.head<3>();
    //  max_p.head<3>() = max_grid * leaf_size_.head<3>();
    min_p = {min_grid[0] * leaf_size_[0], min_grid[1] * leaf_size_[1],
             min_grid[2] * leaf_size_[2], 1.0f};
    max_p = {max_grid[0] * leaf_size_[0], max_grid[1] * leaf_size_[1],
             max_grid[2] * leaf_size_[2], 1.0f};

    // Compute the minimum and maximum bounding box values
    min_b_[0] = static_cast<int64_t>(floor(min_p[0] * inverse_leaf_size_[0]));
    max_b_[0] = static_cast<int64_t>(floor(max_p[0] * inverse_leaf_size_[0]));
    min_b_[1] = static_cast<int64_t>(floor(min_p[1] * inverse_leaf_size_[1]));
    max_b_[1] = static_cast<int64_t>(floor(max_p[1] * inverse_leaf_size_[1]));
    min_b_[2] = static_cast<int64_t>(floor(min_p[2] * inverse_leaf_size_[2]));
    max_b_[2] = static_cast<int64_t>(floor(max_p[2] * inverse_leaf_size_[2]));

    // Compute the number of divisions needed along all axis
    div_b_ = max_b_ - min_b_ + Eigen::Vector4l::Ones();
    div_b_[3] = 0;

    // Set up the division multiplier
    divb_mul_ = Eigen::Vector4l(1, div_b_[0], div_b_[0] * div_b_[1], 0);

    // Storage for mapping leaf and pointcloud indexes
    std::vector<cloud_point_index_idx_ul> index_vector;
    index_vector.reserve(indices_->size());

    // If we don't want to process the entire cloud, but rather filter points far
    // away from the viewpoint first...
    if (!filter_field_name_.empty()) {
        //    // Get the distance field index
        //    std::vector<pcl::PCLPointField> fields;
        //    int distance_idx = pcl::getFieldIndex (*input_, filter_field_name_,
        //    fields);
        //    if (distance_idx == -1)
        //      PCL_WARN ("[pcl::%s::applyFilter] Invalid filter field name. Index
        //      is %d.\n", getClassName ().c_str (), distance_idx);
        //
        //    // First pass: go over all points and insert them into the
        //    index_vector vector
        //    // with calculated idx. Points with the same idx value will contribute
        //    to the
        //    // same point of resulting CloudPoint
        //    for (std::vector<int>::const_iterator it = indices_->begin (); it !=
        //    indices_->end (); ++it)
        //    {
        //      if (!input_->is_dense)
        //        // Check if the point is invalid
        //        if (!pcl_isfinite (input_->points[*it].x) ||
        //            !pcl_isfinite (input_->points[*it].y) ||
        //            !pcl_isfinite (input_->points[*it].z))
        //          continue;
        //
        //      // Get the distance value
        //      const uint8_t* pt_data = reinterpret_cast<const uint8_t*>
        //      (&input_->points[*it]);
        //      float distance_value = 0;
        //      memcpy (&distance_value, pt_data + fields[distance_idx].offset,
        //      sizeof (float));
        //
        //      if (filter_limit_negative_)
        //      {
        //        // Use a threshold for cutting out points which inside the
        //        interval
        //        if ((distance_value < filter_limit_max_) && (distance_value >
        //        filter_limit_min_))
        //          continue;
        //      }
        //      else
        //      {
        //        // Use a threshold for cutting out points which are too close/far
        //        away
        //        if ((distance_value > filter_limit_max_) || (distance_value <
        //        filter_limit_min_))
        //          continue;
        //      }
        //
        //      int64_t ijk0 = static_cast<int64_t> (floor (input_->points[*it].x *
        //      inverse_leaf_size_[0]) - static_cast<float> (min_b_[0]));
        //      int64_t ijk1 = static_cast<int64_t> (floor (input_->points[*it].y *
        //      inverse_leaf_size_[1]) - static_cast<float> (min_b_[1]));
        //      int64_t ijk2 = static_cast<int64_t> (floor (input_->points[*it].z *
        //      inverse_leaf_size_[2]) - static_cast<float> (min_b_[2]));
        //
        //      // Compute the centroid leaf index
        //      int64_t idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 *
        //      divb_mul_[2];
        //      index_vector.push_back (cloud_point_index_idx_ul
        //      (static_cast<int64_t> (idx), *it));
        //    }
    }
        // No distance filtering, process all data
    else {
        // First pass: go over all points and insert them into the index_vector
        // vector
        // with calculated idx. Points with the same idx value will contribute to
        // the
        // same point of resulting CloudPoint
        for (std::vector<int>::const_iterator it = indices_->begin();
             it != indices_->end(); ++it) {
            if (!input_->is_dense)
                // Check if the point is invalid
                if (!pcl_isfinite(input_->points[*it].x) ||
                    !pcl_isfinite(input_->points[*it].y) ||
                    !pcl_isfinite(input_->points[*it].z))
                    continue;

            int64_t ijk0 = static_cast<int64_t>(
                    floor(input_->points[*it].x * inverse_leaf_size_[0]) -
                    static_cast<float>(min_b_[0]));
            int64_t ijk1 = static_cast<int64_t>(
                    floor(input_->points[*it].y * inverse_leaf_size_[1]) -
                    static_cast<float>(min_b_[1]));
            int64_t ijk2 = static_cast<int64_t>(
                    floor(input_->points[*it].z * inverse_leaf_size_[2]) -
                    static_cast<float>(min_b_[2]));

            // Compute the centroid leaf index
            int64_t idx =
                    ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
            index_vector.push_back(
                    cloud_point_index_idx_ul(static_cast<int64_t>(idx), *it));
        }
    }

    // Second pass: sort the index_vector vector using value representing target
    // cell as index
    // in effect all points belonging to the same output cell will be next to each
    // other
    std::sort(index_vector.begin(), index_vector.end(),
              std::less<cloud_point_index_idx_ul>());

    // Third pass: count output cells
    // we need to skip all the same, adjacent idx values
    int64_t total = 0;
    int64_t index = 0;
    // first_and_last_indices_vector[i] represents the index in index_vector of
    // the first point in
    // index_vector belonging to the voxel which corresponds to the i-th output
    // point,
    // and of the first point not belonging to.
    std::vector<std::pair<int64_t, int64_t>> first_and_last_indices_vector;
    // Worst case size
    first_and_last_indices_vector.reserve(index_vector.size());
    while (index < index_vector.size()) {
        int64_t i = index + 1;
        while (i < index_vector.size() &&
               index_vector[i].idx == index_vector[index].idx)
            ++i;
        if (i - index >= min_points_per_voxel_) {
            ++total;
            first_and_last_indices_vector.push_back(
                    std::pair<int64_t, int64_t>(index, i));
        }
        index = i;
    }

    // Fourth pass: compute centroids, insert them into their final position
    if (save_leaf_layout_) {
        try {
            // Resizing won't reset old elements to -1.  If leaf_layout_ has been used
            // previously, it needs to be re-initialized to -1
            int64_t new_layout_size = div_b_[0] * div_b_[1] * div_b_[2];
            // This is the number of elements that need to be re-initialized to -1
            int64_t reinit_size = std::min(static_cast<int64_t>(new_layout_size),
                                           static_cast<int64_t>(leaf_layout_.size()));
            for (int64_t i = 0; i < reinit_size; i++) {
                leaf_layout_[i] = -1;
            }
            leaf_layout_.resize(new_layout_size, -1);
        } catch (std::bad_alloc &) {
            throw PCLException(
                    "VoxelGridIndices bin size is too low; impossible to allocate memory "
                            "for layout",
                    "voxel_grid.hpp", "applyFilter");
        } catch (std::length_error &) {
            throw PCLException(
                    "VoxelGridIndices bin size is too low; impossible to allocate memory "
                            "for layout",
                    "voxel_grid.hpp", "applyFilter");
        }
    }

    index = 0;
    for (size_t cp = 0; cp < first_and_last_indices_vector.size(); ++cp) {
        // calculate centroid - sum values from all input points, that have the same
        // idx value in index_vector array
        int64_t first_index = first_and_last_indices_vector[cp].first;
        int64_t last_index = first_and_last_indices_vector[cp].second;

        // index is centroid final position in resulting PointCloud
        if (save_leaf_layout_) leaf_layout_[index_vector[first_index].idx] = index;

        Eigen::Vector4f centroid(Eigen::Vector4f::Zero());

        for (int64_t li = first_index; li < last_index; ++li)
            centroid +=
                    input_->points[index_vector[li].cloud_point_index].getVector4fMap();

        centroid /= static_cast<float>(last_index - first_index);

        // Find the pointi in voxel closest to centroid
        float dist2 = std::numeric_limits<float>::max();
        int64_t min_idx;

        for (int64_t li = first_index; li < last_index; ++li) {
            Eigen::Vector4f dd =
                    input_->points[index_vector[li].cloud_point_index].getVector4fMap() -
                    centroid;
            float d = (dd.head<3>()).squaredNorm();
            if (d < dist2) {
                dist2 = d;
                min_idx = li;
            }
        }

        indices.push_back(index_vector[min_idx].cloud_point_index);
        if (extract_removed_indices_) {
            for (int64_t li = first_index; li < last_index; ++li) {
                if (li != min_idx)
                    (*removed_indices_).push_back(index_vector[li].cloud_point_index);
            }
        }
        ++index;
    }

    indices.shrink_to_fit();
    removed_indices_->shrink_to_fit();
}

#define PCL_INSTANTIATE_VoxelGridIndices(T) \
  template class PCL_EXPORTS pcl::VoxelGridIndices<T>;