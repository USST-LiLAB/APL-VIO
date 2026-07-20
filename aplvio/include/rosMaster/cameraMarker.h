#ifndef __CAMERA_MARKER_H
#define __CAMERA_MARKER_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>


class CameraMarker {
public:
	CameraMarker(float r, float g, float b, float a = 1.0);
	
	inline void setImageBoundaryColor(float r, float g, float b, float a = 1.0);
	inline void setOpticalCenterConnectorColor(float r, float g, float b, float a = 1.0);
	inline void setScale(double s);
	inline void setLineWidth(double width);
	inline void reset();

	void createMarker(const Eigen::Vector3d &, const Eigen::Quaterniond &);

	static void Eigen2Point(const Eigen::Vector3d &, geometry_msgs::Point &);

	std::vector<visualization_msgs::Marker> m_markers;

private:
	std_msgs::ColorRGBA m_image_boundary_color;
	std_msgs::ColorRGBA m_optical_center_connector_color;
	
	double m_scale;
	double m_line_width;

	static const Eigen::Vector3d imlt;
	static const Eigen::Vector3d imlb;
	static const Eigen::Vector3d imrt;
	static const Eigen::Vector3d imrb;
	static const Eigen::Vector3d oc  ;
	static const Eigen::Vector3d lt0 ;
	static const Eigen::Vector3d lt1 ;
	static const Eigen::Vector3d lt2 ;
};

inline void CameraMarker::setImageBoundaryColor(float r, float g, float b, float a)
{
    m_image_boundary_color.r = r;
    m_image_boundary_color.g = g;
    m_image_boundary_color.b = b;
    m_image_boundary_color.a = a;
}

inline void CameraMarker::setOpticalCenterConnectorColor(float r, float g, float b, float a)
{
    m_optical_center_connector_color.r = r;
    m_optical_center_connector_color.g = g;
    m_optical_center_connector_color.b = b;
    m_optical_center_connector_color.a = a;
}

inline void CameraMarker::setScale(double scale)
{
    m_scale = scale;
}

inline void CameraMarker::setLineWidth(double width)
{
    m_line_width = width;
}

inline void CameraMarker::reset()
{
	m_markers.clear();
}

#endif
