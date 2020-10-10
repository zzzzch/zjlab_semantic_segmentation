#pragma once

#include <pcl/filters/filter_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <map>

namespace Eigen {

    using Vector4l = Eigen::Matrix<int64_t, 4, 1>;
    using Vector3l = Eigen::Matrix<int64_t, 3, 1>;
    using Array4l = Eigen::Array<int64_t, 4, 1>;
    using Array3l = Eigen::Array<int64_t, 3, 1>;
}

namespace pcl {

/** \brief VoxelGrid assembles a local 3D grid over a given PointCloud, and
 * downsamples + filters the data.
  *
  * The VoxelGrid class creates a *3D voxel grid* (think about a voxel
  * grid as a set of tiny 3D boxes in space) over the input point cloud data.
  * Then, in each *voxel* (i.e., 3D box), all the points present will be
  * approximated (i.e., *downsampled*) with their centroid. This approach is
  * a bit slower than approximating them with the center of the voxel, but it
  * represents the underlying surface more accurately.
  *
  * \author Radu B. Rusu, Bastian Steder
  * \ingroup filters
  */
    template <typename PointT>
    class VoxelGridIndices : public FilterIndices<PointT> {
    protected:
        using PCLBase<PointT>::input_;
        using PCLBase<PointT>::indices_;
        using Filter<PointT>::filter_name_;
        using Filter<PointT>::getClassName;
        using FilterIndices<PointT>::negative_;
        using FilterIndices<PointT>::keep_organized_;
        using FilterIndices<PointT>::user_filter_value_;
        using FilterIndices<PointT>::extract_removed_indices_;
        using FilterIndices<PointT>::removed_indices_;

        typedef typename FilterIndices<PointT>::PointCloud PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;
        typedef typename PointCloud::ConstPtr PointCloudConstPtr;

    public:
        typedef boost::shared_ptr<VoxelGridIndices<PointT>> Ptr;
        typedef boost::shared_ptr<const VoxelGridIndices<PointT>> ConstPtr;

        /** \brief Empty constructor. */
        VoxelGridIndices()
                : leaf_size_(Eigen::Vector4f::Zero()),
                  inverse_leaf_size_(Eigen::Array4f::Zero()),
                  save_leaf_layout_(false),
                  leaf_layout_(),
                  min_b_(Eigen::Vector4l::Zero()),
                  max_b_(Eigen::Vector4l::Zero()),
                  div_b_(Eigen::Vector4l::Zero()),
                  divb_mul_(Eigen::Vector4l::Zero()),
                  filter_field_name_(""),
                  filter_limit_min_(-FLT_MAX),
                  filter_limit_max_(FLT_MAX),
                  filter_limit_negative_(false),
                  min_points_per_voxel_(0) {
            filter_name_ = "VoxelGridIndices";
        }

        /** \brief Destructor. */
        virtual ~VoxelGridIndices() {}

        /** \brief Set the voxel grid leaf size.
          * \param[in] leaf_size the voxel grid leaf size
          */
        inline void setLeafSize(const Eigen::Vector4f &leaf_size) {
            leaf_size_ = leaf_size;
            // Avoid division errors
            if (leaf_size_[3] == 0) leaf_size_[3] = 1;
            // Use multiplications instead of divisions
            inverse_leaf_size_ = Eigen::Array4f::Ones() / leaf_size_.array();
        }

        /** \brief Set the voxel grid leaf size.
          * \param[in] lx the leaf size for X
          * \param[in] ly the leaf size for Y
          * \param[in] lz the leaf size for Z
          */
        inline void setLeafSize(float lx, float ly, float lz) {
            leaf_size_[0] = lx;
            leaf_size_[1] = ly;
            leaf_size_[2] = lz;
            // Avoid division errors
            if (leaf_size_[3] == 0) leaf_size_[3] = 1;
            // Use multiplications instead of divisions
            inverse_leaf_size_ = Eigen::Array4f::Ones() / leaf_size_.array();
        }

        /** \brief Get the voxel grid leaf size. */
        inline Eigen::Vector3f getLeafSize() const { return (leaf_size_.head<3>()); }

        /** \brief Set the minimum number of points required for a voxel to be used.
          * \param[in] min_points_per_voxel the minimum number of points for required
         * for a voxel to be used
          */
        inline void setMinimumPointsNumberPerVoxel(
                unsigned int min_points_per_voxel) {
            min_points_per_voxel_ = min_points_per_voxel;
        }

        /** \brief Return the minimum number of points required for a voxel to be
         * used.
         */
        inline unsigned int getMinimumPointsNumberPerVoxel() const {
            return min_points_per_voxel_;
        }

        /** \brief Set to true if leaf layout information needs to be saved for later
         * access.
          * \param[in] save_leaf_layout the new value (true/false)
          */
        inline void setSaveLeafLayout(bool save_leaf_layout) {
            save_leaf_layout_ = save_leaf_layout;
        }

        /** \brief Returns true if leaf layout information will to be saved for later
         * access. */
        inline bool getSaveLeafLayout() const { return (save_leaf_layout_); }

        /** \brief Get the minimum coordinates of the bounding box (after
          * filtering is performed).
          */
        inline Eigen::Vector3l getMinBoxCoordinates() const {
            return (min_b_.head<3>());
        }

        /** \brief Get the minimum coordinates of the bounding box (after
          * filtering is performed).
          */
        inline Eigen::Vector3l getMaxBoxCoordinates() const {
            return (max_b_.head<3>());
        }

        /** \brief Get the number of divisions along all 3 axes (after filtering
          * is performed).
          */
        inline Eigen::Vector3l getNrDivisions() const { return (div_b_.head<3>()); }

        /** \brief Get the multipliers to be applied to the grid coordinates in
          * order to find the centroid index (after filtering is performed).
          */
        inline Eigen::Vector3l getDivisionMultiplier() const {
            return (divb_mul_.head<3>());
        }

        /** \brief Returns the index in the resulting downsampled cloud of the
         * specified point.
          *
          * \note for efficiency, user must make sure that the saving of the leaf
         * layout is enabled and filtering
          * performed, and that the point is inside the grid, to avoid invalid access
         * (or use
          * getGridCoordinates+getCentroidIndexAt)
          *
          * \param[in] p the point to get the index at
          */
        inline int64_t getCentroidIndex(const PointT &p) const {
            return (leaf_layout_.at(
                    (Eigen::Vector4l(
                            static_cast<int64_t>(floor(p.x * inverse_leaf_size_[0])),
                            static_cast<int64_t>(floor(p.y * inverse_leaf_size_[1])),
                            static_cast<int64_t>(floor(p.z * inverse_leaf_size_[2])), 0) -
                     min_b_)
                            .dot(divb_mul_)));
        }

        /** \brief Returns the indices in the resulting downsampled cloud of the
         * points at the specified grid coordinates,
          * relative to the grid coordinates of the specified point (or -1 if the cell
         * was empty/out of bounds).
          * \param[in] reference_point the coordinates of the reference point
         * (corresponding cell is allowed to be empty/out of bounds)
          * \param[in] relative_coordinates matrix with the columns being the
         * coordinates of the requested cells, relative to the reference point's cell
          * \note for efficiency, user must make sure that the saving of the leaf
         * layout is enabled and filtering performed
          */
        inline std::vector<size_t> getNeighborCentroidIndices(
                const PointT &reference_point,
                const Eigen::MatrixXi &relative_coordinates) const {
            Eigen::Vector4l ijk(
                    static_cast<size_t>(floor(reference_point.x * inverse_leaf_size_[0])),
                    static_cast<size_t>(floor(reference_point.y * inverse_leaf_size_[1])),
                    static_cast<size_t>(floor(reference_point.z * inverse_leaf_size_[2])),
                    0);
            Eigen::Array4l diff2min = min_b_ - ijk;
            Eigen::Array4l diff2max = max_b_ - ijk;
            std::vector<size_t> neighbors(relative_coordinates.cols());
            for (size_t ni = 0; ni < relative_coordinates.cols(); ni++) {
                Eigen::Vector4l displacement =
                        (Eigen::Vector4l() << relative_coordinates.col(ni), 0).finished();
                // checking if the specified cell is in the grid
                if ((diff2min <= displacement.array()).all() &&
                    (diff2max >= displacement.array()).all())
                    neighbors[ni] =
                            leaf_layout_[((ijk + displacement - min_b_)
                                    .dot(divb_mul_))];  // .at() can be omitted
                else
                    neighbors[ni] = -1;  // cell is out of bounds, consider it empty
            }
            return (neighbors);
        }

        /** \brief Returns the layout of the leafs for fast access to cells relative
         * to current position.
          * \note position at (i-min_x) + (j-min_y)*div_x + (k-min_z)*div_x*div_y
         * holds the index of the element at coordinates (i,j,k) in the grid (-1 if
         * empty)
          */
        inline std::vector<int64_t> getLeafLayout() const { return (leaf_layout_); }

        /** \brief Returns the corresponding (i,j,k) coordinates in the grid of point
         * (x,y,z).
          * \param[in] x the X point coordinate to get the (i, j, k) index at
          * \param[in] y the Y point coordinate to get the (i, j, k) index at
          * \param[in] z the Z point coordinate to get the (i, j, k) index at
          */
        inline Eigen::Vector3l getGridCoordinates(float x, float y, float z) const {
            return (Eigen::Vector3l(
                    static_cast<int64_t>(floor(x * inverse_leaf_size_[0])),
                    static_cast<int64_t>(floor(y * inverse_leaf_size_[1])),
                    static_cast<int64_t>(floor(z * inverse_leaf_size_[2]))));
        }

        /** \brief Returns the index in the downsampled cloud corresponding to a given
         * set of coordinates.
          * \param[in] ijk the coordinates (i,j,k) in the grid (-1 if empty)
          */
        inline size_t getCentroidIndexAt(const Eigen::Vector3l &ijk) const {
            size_t idx =
                    ((Eigen::Vector4l() << ijk, 0).finished() - min_b_).dot(divb_mul_);
            if (idx >= static_cast<int64_t>(leaf_layout_.size()))  // this checks also
                // if
                // leaf_layout_.size
                // () == 0 i.e.
                // everything was
                // computed as needed
            {
                // if (verbose)
                //  PCL_ERROR ("[pcl::%s::getCentroidIndexAt] Specified coordinate is
                //  outside grid bounds, or leaf layout is not saved, make sure to call
                //  setSaveLeafLayout(true) and filter(output) first!\n", getClassName
                //  ().c_str ());
                return (-1);
            }
            return (leaf_layout_[idx]);
        }

        /** \brief Provide the name of the field to be used for filtering data. In
         * conjunction with  \a setFilterLimits,
          * points having values outside this interval will be discarded.
          * \param[in] field_name the name of the field that contains values used for
         * filtering
          */
        inline void setFilterFieldName(const std::string &field_name) {
            filter_field_name_ = field_name;
        }

        /** \brief Get the name of the field used for filtering. */
        inline std::string const getFilterFieldName() const {
            return (filter_field_name_);
        }

        /** \brief Set the field filter limits. All points having field values outside
         * this interval will be discarded.
          * \param[in] limit_min the minimum allowed field value
          * \param[in] limit_max the maximum allowed field value
          */
        inline void setFilterLimits(const double &limit_min,
                                    const double &limit_max) {
            filter_limit_min_ = limit_min;
            filter_limit_max_ = limit_max;
        }

        /** \brief Get the field filter limits (min/max) set by the user. The default
         * values are -FLT_MAX, FLT_MAX.
          * \param[out] limit_min the minimum allowed field value
          * \param[out] limit_max the maximum allowed field value
          */
        inline void getFilterLimits(double &limit_min, double &limit_max) const {
            limit_min = filter_limit_min_;
            limit_max = filter_limit_max_;
        }

        /** \brief Set to true if we want to return the data outside the interval
         * specified by setFilterLimits (min, max).
          * Default: false.
          * \param[in] limit_negative return data inside the interval (false) or
         * outside (true)
          */
        inline void setFilterLimitsNegative(const bool limit_negative) {
            filter_limit_negative_ = limit_negative;
        }

        /** \brief Get whether the data outside the interval (min/max) is to be
         * returned (true) or inside (false).
          * \param[out] limit_negative true if data \b outside the interval [min; max]
         * is to be returned, false otherwise
          */
        inline void getFilterLimitsNegative(bool &limit_negative) const {
            limit_negative = filter_limit_negative_;
        }

        /** \brief Get whether the data outside the interval (min/max) is to be
         * returned (true) or inside (false).
          * \return true if data \b outside the interval [min; max] is to be returned,
         * false otherwise
          */
        inline bool getFilterLimitsNegative() const {
            return (filter_limit_negative_);
        }

        inline bool getExtractRemovedIndices() { return extract_removed_indices_; }

        inline void setExtractRemovedIndices(bool extract_removed_indices) {
            extract_removed_indices_ = extract_removed_indices;
        }

    protected:
        /** \brief The size of a leaf. */
        Eigen::Vector4f leaf_size_;

        /** \brief Internal leaf sizes stored as 1/leaf_size_ for efficiency reasons.
         */
        Eigen::Array4f inverse_leaf_size_;

        /** \brief Set to true if leaf layout information needs to be saved in \a
         * leaf_layout_. */
        bool save_leaf_layout_;

        /** \brief The leaf layout information for fast access to cells relative to
         * current position **/
        std::vector<int64_t> leaf_layout_;

        /** \brief The minimum and maximum bin coordinates, the number of divisions,
         * and the division multiplier. */
        Eigen::Vector4l min_b_, max_b_, div_b_, divb_mul_;

        /** \brief The desired user filter field name. */
        std::string filter_field_name_;

        /** \brief The minimum allowed filter value a point will be considered from.
         */
        double filter_limit_min_;

        /** \brief The maximum allowed filter value a point will be considered from.
         */
        double filter_limit_max_;

        /** \brief Set to true if we want to return the data outside (\a
         * filter_limit_min_;\a filter_limit_max_). Default: false. */
        bool filter_limit_negative_;

        /** \brief Minimum number of points per voxel for the centroid to be computed
         */
        unsigned int min_points_per_voxel_;

        typedef typename pcl::traits::fieldList<PointT>::type FieldList;

//        /** \brief Filtered results are stored in a separate point cloud.
//          * \param[out] output The resultant point cloud.
//          */
//        void applyFilter(PointCloud &output);
//
//        /** \brief Filtered results are indexed by an indices array.
//          * \param[out] indices The resultant indices.
//          */
//        void applyFilter(std::vector<int> &indices) { applyFilterIndices(indices); }
//
//        /** \brief Filtered results are indexed by an indices array.
//          * \param[out] indices The resultant indices.
//          */
//        void applyFilterIndices(std::vector<int> &indices);
    };
}


