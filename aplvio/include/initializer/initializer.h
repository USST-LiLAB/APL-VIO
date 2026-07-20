#ifndef __INITIALIZER_H
#define __INITIALIZER_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "estimator/customType.h"
#include "visual/featureManager.h"
#include "inertial/imuProcessor.h"


class Estimator;

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		: observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0];
		p[1] += camera_T[1];
		p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  	return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3> 
			(new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};


class Initializer
{
public:
    Initializer();
    void clearState();
    void setParameters();

    bool initialStructure(ImageFrameList &, FrameWindow &, std::shared_ptr<FeatureManager>);

	bool construct();
	bool optimization();
	bool reconstruct(ImageFrameList &, FrameWindow &);
	bool visualInertialAlignment(ImageFrameList &, FrameWindow &);
	bool refineGravity(ImageFrameList &);
	bool recoveryStructure(ImageFrameList &, FrameWindow &, std::shared_ptr<FeatureManager>);

    bool selectReferanceFrame(std::shared_ptr<FeatureManager>);
    bool solveRelativeRt(const FeaturePairs &, Eigen::Matrix3d &, Eigen::Vector3d &);
    bool solveFrameByPnP(Eigen::Matrix3d &, Eigen::Vector3d &, int);
    void solveGyrBias(ImageFrameList &, FrameWindow &);
    void triangulateTwoFrames(int, Eigen::Matrix<double, 3, 4> &, int, Eigen::Matrix<double, 3, 4> &);
	Eigen::MatrixXd tangentBasis(Eigen::Vector3d &);

	double initial_timestamp = 0;

private:
	Eigen::VectorXd x;
	double s; // scale
	const double scale_power = 100.0;
	Eigen::Vector3d g; // gravity
	int ref_count;

    Eigen::Quaterniond Q[WINDOW_SIZE + 1];
    Eigen::Matrix3d R[WINDOW_SIZE + 1];
    Eigen::Vector3d P[WINDOW_SIZE + 1];
    // converted pose
	Eigen::Quaterniond cQ[WINDOW_SIZE + 1];
	Eigen::Matrix3d cR[WINDOW_SIZE + 1];
	Eigen::Vector3d cP[WINDOW_SIZE + 1];
	Eigen::Matrix<double, 3, 4> cT[WINDOW_SIZE + 1];

    std::vector<SFMFeature> sfm_features;
};

#endif
