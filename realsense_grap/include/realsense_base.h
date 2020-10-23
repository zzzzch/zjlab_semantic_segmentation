// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once

//#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <librealsense2/rs.hpp>
#include <imgui.h>
#include "imgui_impl_glfw.h"

#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <map>

#define PI 3.14159265358979323846
#define IMU_FRAME_WIDTH 1280
#define IMU_FRAME_HEIGHT 720
//////////////////////////////
// Basic Data Types         //
//////////////////////////////

struct float3 { float x, y, z; };
struct float2 { float x, y; };

struct rect
{
    float x, y;
    float w, h;

    // Create new rect within original boundaries with give aspect ration
    rect adjust_ratio(float2 size) const
    {
        auto H = static_cast<float>(h), W = static_cast<float>(h) * size.x / size.y;
        if (W > w)
        {
            auto scale = w / W;
            W *= scale;
            H *= scale;
        }

        return{ x + (w - W) / 2, y + (h - H) / 2, W, H };
    }
};

//////////////////////////////
// Simple font loading code //
//////////////////////////////

#include "stb_easy_font.h"

inline void draw_text(int x, int y, const char * text)
{
    char buffer[60000]; // ~300 chars
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 16, buffer);
    glDrawArrays(GL_QUADS, 0, 4 * stb_easy_font_print((float)x, (float)(y - 7), (char *)text, nullptr, buffer, sizeof(buffer)));
    glDisableClientState(GL_VERTEX_ARRAY);
}

void set_viewport(const rect& r)
{
    glViewport( (int)r.x, (int)r.y, (int)r.w, (int)r.h);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glOrtho(0, r.w, r.h, 0, -1, +1);
}

class imu_renderer
{
public:
    void render(const rs2::motion_frame& frame, const rect& r)
    {
        draw_motion(frame, r.adjust_ratio({ IMU_FRAME_WIDTH, IMU_FRAME_HEIGHT }));
    }

    GLuint get_gl_handle() { return _gl_handle; }

private:
    GLuint _gl_handle = 0;

    void draw_motion(const rs2::motion_frame& f, const rect& r)
    {
        if (!_gl_handle)
            glGenTextures(1, &_gl_handle);

        set_viewport(r);
        draw_text(int(0.05f * r.w), int(0.05f * r.h), f.get_profile().stream_name().c_str());

        auto md = f.get_motion_data();
        auto x = md.x;
        auto y = md.y;
        auto z = md.z;

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        glOrtho(-2.8, 2.8, -2.4, 2.4, -7, 7);

        glRotatef(25, 1.0f, 0.0f, 0.0f);

        glTranslatef(0, -0.33f, -1.f);

        glRotatef(-135, 0.0f, 1.0f, 0.0f);

        glRotatef(180, 0.0f, 0.0f, 1.0f);
        glRotatef(-90, 0.0f, 1.0f, 0.0f);

        draw_axes(1,2);

        draw_circle(1, 0, 0, 0, 1, 0);
        draw_circle(0, 1, 0, 0, 0, 1);
        draw_circle(1, 0, 0, 0, 0, 1);

        const auto canvas_size = 230;
        const auto vec_threshold = 0.01f;
        float norm = std::sqrt(x * x + y * y + z * z);
        if (norm < vec_threshold)
        {
            const auto radius = 0.05;
            static const int circle_points = 100;
            static const float angle = 2.0f * 3.1416f / circle_points;

            glColor3f(1.0f, 1.0f, 1.0f);
            glBegin(GL_POLYGON);
            double angle1 = 0.0;
            glVertex2d(radius * cos(0.0), radius * sin(0.0));
            int i;
            for (i = 0; i < circle_points; i++)
            {
                glVertex2d(radius * cos(angle1), radius *sin(angle1));
                angle1 += angle;
            }
            glEnd();
        }
        else
        {
            auto vectorWidth = 3.f;
            glLineWidth(vectorWidth);
            glBegin(GL_LINES);
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);
            glVertex3f(x / norm, y / norm, z / norm);
            glEnd();

            // Save model and projection matrix for later
            GLfloat model[16];
            glGetFloatv(GL_MODELVIEW_MATRIX, model);
            GLfloat proj[16];
            glGetFloatv(GL_PROJECTION_MATRIX, proj);

            glLoadIdentity();
            glOrtho(-canvas_size, canvas_size, -canvas_size, canvas_size, -1, +1);

            std::ostringstream s1;
            const auto precision = 3;

            glRotatef(180, 1.0f, 0.0f, 0.0f);

            s1 << "(" << std::fixed << std::setprecision(precision) << x << "," << std::fixed << std::setprecision(precision) << y << "," << std::fixed << std::setprecision(precision) << z << ")";
            print_text_in_3d(x, y, z, s1.str().c_str(), false, model, proj, 1 / norm);

            std::ostringstream s2;
            s2 << std::setprecision(precision) << norm;
            print_text_in_3d(x / 2, y / 2, z / 2, s2.str().c_str(), true, model, proj, 1 / norm);
        }
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    //IMU drawing helper functions
    void multiply_vector_by_matrix(GLfloat vec[], GLfloat mat[], GLfloat* result)
    {
        const auto N = 4;
        for (int i = 0; i < N; i++)
        {
            result[i] = 0;
            for (int j = 0; j < N; j++)
            {
                result[i] += vec[j] * mat[N*j + i];
            }
        }
        return;
    }

    float2 xyz_to_xy(float x, float y, float z, GLfloat model[], GLfloat proj[], float vec_norm)
    {
        GLfloat vec[4] = { x, y, z, 0 };
        float tmp_result[4];
        float result[4];

        const auto canvas_size = 230;

        multiply_vector_by_matrix(vec, model, tmp_result);
        multiply_vector_by_matrix(tmp_result, proj, result);

        return{ canvas_size * vec_norm *result[0], canvas_size * vec_norm *result[1] };
    }

    void print_text_in_3d(float x, float y, float z, const char* text, bool center_text, GLfloat model[], GLfloat proj[], float vec_norm)
    {
        auto xy = xyz_to_xy(x, y, z, model, proj, vec_norm);
        auto w = (center_text) ? stb_easy_font_width((char*)text) : 0;
        glColor3f(1.0f, 1.0f, 1.0f);
        draw_text((int)(xy.x - w / 2), (int)xy.y, text);
    }

    static void  draw_axes(float axis_size = 1.f, float axisWidth = 4.f)
    {
        // Triangles For X axis
        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(axis_size * 1.1f, 0.f, 0.f);
        glVertex3f(axis_size, -axis_size * 0.05f, 0.f);
        glVertex3f(axis_size, axis_size * 0.05f, 0.f);
        glVertex3f(axis_size * 1.1f, 0.f, 0.f);
        glVertex3f(axis_size, 0.f, -axis_size * 0.05f);
        glVertex3f(axis_size, 0.f, axis_size * 0.05f);
        glEnd();

        // Triangles For Y axis
        glBegin(GL_TRIANGLES);
        glColor3f(0.f, 1.f, 0.f);
        glVertex3f(0.f, axis_size * 1.1f, 0.0f);
        glVertex3f(0.f, axis_size, 0.05f * axis_size);
        glVertex3f(0.f, axis_size, -0.05f * axis_size);
        glVertex3f(0.f, axis_size * 1.1f, 0.0f);
        glVertex3f(0.05f * axis_size, axis_size, 0.f);
        glVertex3f(-0.05f * axis_size, axis_size, 0.f);
        glEnd();

        // Triangles For Z axis
        glBegin(GL_TRIANGLES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 1.1f * axis_size);
        glVertex3f(0.0f, 0.05f * axis_size, 1.0f * axis_size);
        glVertex3f(0.0f, -0.05f * axis_size, 1.0f * axis_size);
        glVertex3f(0.0f, 0.0f, 1.1f * axis_size);
        glVertex3f(0.05f * axis_size, 0.f, 1.0f * axis_size);
        glVertex3f(-0.05f * axis_size, 0.f, 1.0f * axis_size);
        glEnd();

        glLineWidth(axisWidth);

        // Drawing Axis
        glBegin(GL_LINES);
        // X axis - Red
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(axis_size, 0.0f, 0.0f);

        // Y axis - Green
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, axis_size, 0.0f);

        // Z axis - Blue
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, axis_size);
        glEnd();
    }

    // intensity is grey intensity
    static void draw_circle(float xx, float xy, float xz, float yx, float yy, float yz, float radius = 1.1, float3 center = { 0.0, 0.0, 0.0 }, float intensity = 0.5f)
    {
        const auto N = 50;
        glColor3f(intensity, intensity, intensity);
        glLineWidth(2);
        glBegin(GL_LINE_STRIP);

        for (int i = 0; i <= N; i++)
        {
            const double theta = (2 * PI / N) * i;
            const auto cost = static_cast<float>(cos(theta));
            const auto sint = static_cast<float>(sin(theta));
            glVertex3f(
                    center.x + radius * (xx * cost + yx * sint),
                    center.y + radius * (xy * cost + yy * sint),
                    center.z + radius * (xz * cost + yz * sint)
            );
        }
        glEnd();
    }

};

class pose_renderer
{
public:
    void render(const rs2::pose_frame& frame, const rect& r)
    {
        draw_pose(frame, r.adjust_ratio({ IMU_FRAME_WIDTH, IMU_FRAME_HEIGHT }));
    }

    GLuint get_gl_handle() { return _gl_handle; }

private:
    mutable GLuint _gl_handle = 0;

    // Provide textual representation only
    void draw_pose(const rs2::pose_frame& f, const rect& r)
    {
        if (!_gl_handle)
            glGenTextures(1, &_gl_handle);

        set_viewport(r);
        std::string caption(f.get_profile().stream_name());
        if (f.get_profile().stream_index())
            caption += std::to_string(f.get_profile().stream_index());
        draw_text(int(0.05f * r.w), int(0.05f * r.h), caption.c_str());

        auto pose = f.get_pose_data();
        std::stringstream ss;
        ss << "Pos (meter): \t\t" << std::fixed << std::setprecision(2) << pose.translation.x << ", " << pose.translation.y << ", " << pose.translation.z;
        draw_text(int(0.05f * r.w), int(0.2f*r.h), ss.str().c_str());
        ss.clear(); ss.str("");
        ss << "Orient (quaternion): \t" << pose.rotation.x << ", " << pose.rotation.y << ", " << pose.rotation.z << ", " << pose.rotation.w;
        draw_text(int(0.05f * r.w), int(0.3f*r.h), ss.str().c_str());
        ss.clear(); ss.str("");
        ss << "Lin Velocity (m/sec): \t" << pose.velocity.x << ", " << pose.velocity.y << ", " << pose.velocity.z;
        draw_text(int(0.05f * r.w), int(0.4f*r.h), ss.str().c_str());
        ss.clear(); ss.str("");
        ss << "Ang. Velocity (rad/sec): \t" << pose.angular_velocity.x << ", " << pose.angular_velocity.y << ", " << pose.angular_velocity.z;
        draw_text(int(0.05f * r.w), int(0.5f*r.h), ss.str().c_str());
    }
};

/// \brief Print flat 2D text over openGl window
struct text_renderer
{
    // Provide textual representation only
    void put_text(const std::string& msg, float norm_x_pos, float norm_y_pos, const rect& r)
    {
        set_viewport(r);
        draw_text(int(norm_x_pos * r.w), int(norm_y_pos * r.h), msg.c_str());
    }
};

////////////////////////
// Image display code //
////////////////////////
class texture
{
    GLuint gl_handle = 0;
    rs2_stream stream = RS2_STREAM_ANY;

public:
    void render(const rs2::video_frame& frame, const rect& rect)
    {
        upload(frame);
        show(rect.adjust_ratio({ (float)frame.get_width(), (float)frame.get_height() }));
    }

    void upload(const rs2::video_frame& frame)
    {
        if (!frame) return;

        if (!gl_handle)
            glGenTextures(1, &gl_handle);
        GLenum err = glGetError();

        auto format = frame.get_profile().format();
        auto width = frame.get_width();
        auto height = frame.get_height();
        stream = frame.get_profile().stream_type();

        glBindTexture(GL_TEXTURE_2D, gl_handle);

        switch (format)
        {
            case RS2_FORMAT_RGB8:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.get_data());
                break;
            case RS2_FORMAT_RGBA8:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame.get_data());
                break;
            case RS2_FORMAT_Y8:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, frame.get_data());
                break;
            default:
                throw std::runtime_error("The requested format is not supported by this demo!");
        }

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void show(const rect& r) const
    {
        if (!gl_handle)
            return;

        set_viewport(r);

        glBindTexture(GL_TEXTURE_2D, gl_handle);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(0, 1); glVertex2f(0, r.h);
        glTexCoord2f(1, 1); glVertex2f(r.w, r.h);
        glTexCoord2f(1, 0); glVertex2f(r.w, 0);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        draw_text(int(0.05f * r.w), int(r.h - 0.05f*r.h), rs2_stream_to_string(stream));
    }

    GLuint get_gl_handle() { return gl_handle; }
};

class imu_drawer
{
public:
    void render(const rs2::motion_frame& frame, const rect& r)
    {
        draw_motion(frame, r.adjust_ratio({ IMU_FRAME_WIDTH, IMU_FRAME_HEIGHT }));
    }

    GLuint get_gl_handle() { return _gl_handle; }

private:
    GLuint _gl_handle = 0;

    void draw_motion(const rs2::motion_frame& f, const rect& r)
    {
        if (!_gl_handle)
            glGenTextures(1, &_gl_handle);

        set_viewport(r);
        draw_text(int(0.05f * r.w), int(r.h - 0.1f*r.h), f.get_profile().stream_name().c_str());

        auto md = f.get_motion_data();
        auto x = md.x;
        auto y = md.y;
        auto z = md.z;

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        glOrtho(-2.8, 2.8, -2.4, 2.4, -7, 7);

        glRotatef(25, 1.0f, 0.0f, 0.0f);

        glTranslatef(0, -0.33f, -1.f);

        glRotatef(-135, 0.0f, 1.0f, 0.0f);

        glRotatef(180, 0.0f, 0.0f, 1.0f);
        glRotatef(-90, 0.0f, 1.0f, 0.0f);

        draw_axes();

        draw_circle(1, 0, 0, 0, 1, 0);
        draw_circle(0, 1, 0, 0, 0, 1);
        draw_circle(1, 0, 0, 0, 0, 1);

        const auto canvas_size = 230;
        const auto vec_threshold = 0.01f;
        float norm = std::sqrt(x * x + y * y + z * z);
        if (norm < vec_threshold)
        {
            const auto radius = 0.05;
            static const int circle_points = 100;
            static const float angle = 2.0f * 3.1416f / circle_points;

            glColor3f(1.0f, 1.0f, 1.0f);
            glBegin(GL_POLYGON);
            double angle1 = 0.0;
            glVertex2d(radius * cos(0.0), radius * sin(0.0));
            int i;
            for (i = 0; i < circle_points; i++)
            {
                glVertex2d(radius * cos(angle1), radius *sin(angle1));
                angle1 += angle;
            }
            glEnd();
        }
        else
        {
            auto vectorWidth = 5.f;
            glLineWidth(vectorWidth);
            glBegin(GL_LINES);
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertex3f(0.0f, 0.0f, 0.0f);
            glVertex3f(x / norm, y / norm, z / norm);
            glEnd();

            // Save model and projection matrix for later
            GLfloat model[16];
            glGetFloatv(GL_MODELVIEW_MATRIX, model);
            GLfloat proj[16];
            glGetFloatv(GL_PROJECTION_MATRIX, proj);

            glLoadIdentity();
            glOrtho(-canvas_size, canvas_size, -canvas_size, canvas_size, -1, +1);

            std::ostringstream s1;
            const auto precision = 3;

            glRotatef(180, 1.0f, 0.0f, 0.0f);

            s1 << "(" << std::fixed << std::setprecision(precision) << x << "," << std::fixed << std::setprecision(precision) << y << "," << std::fixed << std::setprecision(precision) << z << ")";
            print_text_in_3d(x, y, z, s1.str().c_str(), false, model, proj, 1 / norm);

            std::ostringstream s2;
            s2 << std::setprecision(precision) << norm;
            print_text_in_3d(x / 2, y / 2, z / 2, s2.str().c_str(), true, model, proj, 1 / norm);
        }
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    //IMU drawing helper functions
    void multiply_vector_by_matrix(GLfloat vec[], GLfloat mat[], GLfloat* result)
    {
        const auto N = 4;
        for (int i = 0; i < N; i++)
        {
            result[i] = 0;
            for (int j = 0; j < N; j++)
            {
                result[i] += vec[j] * mat[N*j + i];
            }
        }
        return;
    }

    float2 xyz_to_xy(float x, float y, float z, GLfloat model[], GLfloat proj[], float vec_norm)
    {
        GLfloat vec[4] = { x, y, z, 0 };
        float tmp_result[4];
        float result[4];

        const auto canvas_size = 230;

        multiply_vector_by_matrix(vec, model, tmp_result);
        multiply_vector_by_matrix(tmp_result, proj, result);

        return{ canvas_size * vec_norm *result[0], canvas_size * vec_norm *result[1] };
    }

    void print_text_in_3d(float x, float y, float z, const char* text, bool center_text, GLfloat model[], GLfloat proj[], float vec_norm)
    {
        auto xy = xyz_to_xy(x, y, z, model, proj, vec_norm);
        auto w = (center_text) ? stb_easy_font_width((char*)text) : 0;

        glColor3f(1.0f, 1.0f, 1.0f);
        draw_text((int)(xy.x - w / 2), (int)xy.y, text);
    }

    static void  draw_axes(float axis_size = 1.f, float axisWidth = 4.f)
    {
        // Triangles For X axis
        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(axis_size * 1.1f, 0.f, 0.f);
        glVertex3f(axis_size, -axis_size * 0.05f, 0.f);
        glVertex3f(axis_size, axis_size * 0.05f, 0.f);
        glVertex3f(axis_size * 1.1f, 0.f, 0.f);
        glVertex3f(axis_size, 0.f, -axis_size * 0.05f);
        glVertex3f(axis_size, 0.f, axis_size * 0.05f);
        glEnd();

        // Triangles For Y axis
        glBegin(GL_TRIANGLES);
        glColor3f(0.f, 1.f, 0.f);
        glVertex3f(0.f, axis_size * 1.1f, 0.0f);
        glVertex3f(0.f, axis_size, 0.05f * axis_size);
        glVertex3f(0.f, axis_size, -0.05f * axis_size);
        glVertex3f(0.f, axis_size * 1.1f, 0.0f);
        glVertex3f(0.05f * axis_size, axis_size, 0.f);
        glVertex3f(-0.05f * axis_size, axis_size, 0.f);
        glEnd();

        // Triangles For Z axis
        glBegin(GL_TRIANGLES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 1.1f * axis_size);
        glVertex3f(0.0f, 0.05f * axis_size, 1.0f * axis_size);
        glVertex3f(0.0f, -0.05f * axis_size, 1.0f * axis_size);
        glVertex3f(0.0f, 0.0f, 1.1f * axis_size);
        glVertex3f(0.05f * axis_size, 0.f, 1.0f * axis_size);
        glVertex3f(-0.05f * axis_size, 0.f, 1.0f * axis_size);
        glEnd();

        glLineWidth(axisWidth);

        // Drawing Axis
        glBegin(GL_LINES);
        // X axis - Red
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(axis_size, 0.0f, 0.0f);

        // Y axis - Green
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, axis_size, 0.0f);

        // Z axis - Blue
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, axis_size);
        glEnd();
    }

    // intensity is grey intensity
    static void draw_circle(float xx, float xy, float xz, float yx, float yy, float yz, float radius = 1.1, float3 center = { 0.0, 0.0, 0.0 }, float intensity = 0.5f)
    {
        const auto N = 50;
        glColor3f(intensity, intensity, intensity);
        glLineWidth(2);
        glBegin(GL_LINE_STRIP);

        for (int i = 0; i <= N; i++)
        {
            const double theta = (2 * PI / N) * i;
            const auto cost = static_cast<float>(cos(theta));
            const auto sint = static_cast<float>(sin(theta));
            glVertex3f(
                    center.x + radius * (xx * cost + yx * sint),
                    center.y + radius * (xy * cost + yy * sint),
                    center.z + radius * (xz * cost + yz * sint)
            );
        }
        glEnd();
    }

};

class window
{
public:
    std::function<void(bool)>           on_left_mouse = [](bool) {};
    std::function<void(double, double)> on_mouse_scroll = [](double, double) {};
    std::function<void(double, double)> on_mouse_move = [](double, double) {};
    std::function<void(int)>            on_key_release = [](int) {};

    window(int width, int height, const char* title)
            : _width(width), _height(height)
    {
        glfwInit();
        win = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!win)
            throw std::runtime_error("Could not open OpenGL window, please check your graphic drivers or use the textual SDK tools");
        glfwMakeContextCurrent(win);

        glfwSetWindowUserPointer(win, this);
        glfwSetMouseButtonCallback(win, [](GLFWwindow * w, int button, int action, int mods)
        {
            auto s = (window*)glfwGetWindowUserPointer(w);
            if (button == 0) s->on_left_mouse(action == GLFW_PRESS);
        });

        glfwSetScrollCallback(win, [](GLFWwindow * w, double xoffset, double yoffset)
        {
            auto s = (window*)glfwGetWindowUserPointer(w);
            s->on_mouse_scroll(xoffset, yoffset);
        });

        glfwSetCursorPosCallback(win, [](GLFWwindow * w, double x, double y)
        {
            auto s = (window*)glfwGetWindowUserPointer(w);
            s->on_mouse_move(x, y);
        });

        glfwSetKeyCallback(win, [](GLFWwindow * w, int key, int scancode, int action, int mods)
        {
            auto s = (window*)glfwGetWindowUserPointer(w);
            if (0 == action) // on key release
            {
                s->on_key_release(key);
            }
        });
    }

    float width() const { return float(_width); }
    float height() const { return float(_height); }

    operator bool()
    {
        glPopMatrix();
        glfwSwapBuffers(win);

        auto res = !glfwWindowShouldClose(win);

        glfwPollEvents();
        glfwGetFramebufferSize(win, &_width, &_height);

        // Clear the framebuffer
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(0, 0, _width, _height);

        // Draw the images
        glPushMatrix();
        glfwGetWindowSize(win, &_width, &_height);
        glOrtho(0, _width, _height, 0, -1, +1);

        return res;
    }

    ~window()
    {
        glfwDestroyWindow(win);
        glfwTerminate();
    }

    void show(rs2::frame frame)
    {
        show(frame, { 0, 0, (float)_width, (float)_height });
    }

    void show(const rs2::frame& frame, const rect& rect)
    {
        if (auto fs = frame.as<rs2::frameset>())
            render_frameset(fs, rect);
        if (auto vf = frame.as<rs2::video_frame>())
            render_video_frame(vf, rect);
        if (auto mf = frame.as<rs2::motion_frame>())
            render_motoin_frame(mf, rect);
    }

    operator GLFWwindow*() { return win; }

private:
    GLFWwindow * win;
    std::map<int, texture> _textures;
    std::map<int, imu_drawer> _imus;
    int _width, _height;

    void render_video_frame(const rs2::video_frame& f, const rect& r)
    {
        auto& t = _textures[f.get_profile().unique_id()];
        t.render(f, r);
    }

    void render_motoin_frame(const rs2::motion_frame& f, const rect& r)
    {
        auto& i = _imus[f.get_profile().unique_id()];
        i.render(f, r);
    }

    void render_frameset(const rs2::frameset& frames, const rect& r)
    {
        std::vector<rs2::frame> supported_frames;
        for (auto f : frames)
        {
            if (can_render(f))
                supported_frames.push_back(f);
        }
        if (supported_frames.empty())
            return;

        std::sort(supported_frames.begin(), supported_frames.end(), [](rs2::frame first, rs2::frame second)
        { return first.get_profile().stream_type() < second.get_profile().stream_type();  });

        auto image_grid = calc_grid(r, supported_frames);

        int image_index = 0;
        for (auto f : supported_frames)
        {
            auto r = image_grid.at(image_index);
            show(f, r);
            image_index++;
        }
    }

    bool can_render(const rs2::frame& f) const
    {
        auto format = f.get_profile().format();
        switch (format)
        {
            case RS2_FORMAT_RGB8:
            case RS2_FORMAT_RGBA8:
            case RS2_FORMAT_Y8:
            case RS2_FORMAT_MOTION_XYZ32F:
                return true;
            default:
                return false;
        }
    }

    rect calc_grid(rect r, size_t streams)
    {
        if (r.w <= 0 || r.h <= 0 || streams <= 0)
            throw std::runtime_error("invalid window configuration request, failed to calculate window grid");
        float ratio = r.w / r.h;
        auto x = sqrt(ratio * (float)streams);
        auto y = (float)streams / x;
        auto w = round(x);
        auto h = round(y);
        if (w == 0 || h == 0)
            throw std::runtime_error("invalid window configuration request, failed to calculate window grid");
        while (w*h > streams)
            h > w ? h-- : w--;
        while (w*h < streams)
            h > w ? w++ : h++;
        auto new_w = round(r.w / w);
        auto new_h = round(r.h / h);
        // column count, line count, cell width cell height
        return rect{ static_cast<float>(w), static_cast<float>(h), static_cast<float>(new_w), static_cast<float>(new_h) };
    }

    std::vector<rect> calc_grid(rect r, std::vector<rs2::frame>& frames)
    {
        auto grid = calc_grid(r, frames.size());

        std::vector<rect> rv;
        int curr_line = -1;

        for (int i = 0; i < frames.size(); i++)
        {
            auto mod = i % (int)grid.x;
            float fw = IMU_FRAME_WIDTH;
            float fh = IMU_FRAME_HEIGHT;
            if (auto vf = frames[i].as<rs2::video_frame>())
            {
                fw = (float)vf.get_width();
                fh = (float)vf.get_height();
            }
            float cell_x_postion = (float)(mod * grid.w);
            if (mod == 0) curr_line++;
            float cell_y_position = curr_line * grid.h;
            float2 margin = { grid.w * 0.02f, grid.h * 0.02f };
            auto r = rect{ cell_x_postion + margin.x, cell_y_position + margin.y, grid.w - 2 * margin.x, grid.h };
            rv.push_back(r.adjust_ratio(float2{ fw, fh }));
        }

        return rv;
    }
};

// Struct to get keys pressed on window
struct window_key_listener {
    int last_key = GLFW_KEY_UNKNOWN;

    window_key_listener(window& win) {
        win.on_key_release = std::bind(&window_key_listener::on_key_release, this, std::placeholders::_1);
    }

    void on_key_release(int key) {
        last_key = key;
    }

    int get_key() {
        int key = last_key;
        last_key = GLFW_KEY_UNKNOWN;
        return key;
    }
};

// Struct for managing rotation of pointcloud view
struct glfw_state {
    glfw_state() : yaw(15.0), pitch(15.0), last_x(0.0), last_y(0.0),
                   ml(false), offset_x(2.f), offset_y(2.f), tex() {}
    double yaw;
    double pitch;
    double last_x;
    double last_y;
    bool ml;
    float offset_x;
    float offset_y;
    texture tex;
};

// Handles all the OpenGL calls needed to display the point cloud
void draw_pointcloud(float width, float height, glfw_state& app_state, rs2::points& points)
{
    if (!points)
        return;

    // OpenGL commands that prep screen for the pointcloud
    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(0, 0, +0.5f + app_state.offset_y*0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    glPointSize(width / 640);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());
    float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
    glBegin(GL_POINTS);


    /* this segment actually prints the pointcloud */
    auto vertices = points.get_vertices();              // get vertices
    auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
    for (int i = 0; i < points.size(); i++)
    {
        if (vertices[i].z)
        {
            // upload the point and texture coordinates only for points we have depth data for
            glVertex3fv(vertices[i]);
            glTexCoord2fv(tex_coords[i]);
        }
    }

    // OpenGL cleanup
    glEnd();
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
}

void quat2mat(rs2_quaternion& q, GLfloat H[16])  // to column-major matrix
{
    H[0] = 1 - 2*q.y*q.y - 2*q.z*q.z; H[4] = 2*q.x*q.y - 2*q.z*q.w;     H[8] = 2*q.x*q.z + 2*q.y*q.w;     H[12] = 0.0f;
    H[1] = 2*q.x*q.y + 2*q.z*q.w;     H[5] = 1 - 2*q.x*q.x - 2*q.z*q.z; H[9] = 2*q.y*q.z - 2*q.x*q.w;     H[13] = 0.0f;
    H[2] = 2*q.x*q.z - 2*q.y*q.w;     H[6] = 2*q.y*q.z + 2*q.x*q.w;     H[10] = 1 - 2*q.x*q.x - 2*q.y*q.y; H[14] = 0.0f;
    H[3] = 0.0f;                      H[7] = 0.0f;                      H[11] = 0.0f;                      H[15] = 1.0f;
}

// Handles all the OpenGL calls needed to display the point cloud w.r.t. static reference frame
void draw_pointcloud_wrt_world(float width, float height, glfw_state& app_state, rs2::points& points, rs2_pose& pose, float H_t265_d400[16], std::vector<rs2_vector>& trajectory)
{
    if (!points)
        return;

    // OpenGL commands that prep screen for the pointcloud
    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);


    // viewing matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // rotated from depth to world frame: z => -z, y => -y
    glTranslatef(0, 0, -0.75f-app_state.offset_y*0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, -1, 0);
    glTranslatef(0, 0, 0.5f);

    // draw trajectory
    glEnable(GL_DEPTH_TEST);
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (auto&& v : trajectory)
    {
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(v.x, v.y, v.z);
    }
    glEnd();
    glLineWidth(0.5f);
    glColor3f(1.0f, 1.0f, 1.0f);

    // T265 pose
    GLfloat H_world_t265[16];
    quat2mat(pose.rotation, H_world_t265);
    H_world_t265[12] = pose.translation.x;
    H_world_t265[13] = pose.translation.y;
    H_world_t265[14] = pose.translation.z;

    glMultMatrixf(H_world_t265);

    // T265 to D4xx extrinsics
    glMultMatrixf(H_t265_d400);


    glPointSize(width / 640);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());
    float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
    glBegin(GL_POINTS);

    /* this segment actually prints the pointcloud */
    auto vertices = points.get_vertices();              // get vertices
    auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
    for (int i = 0; i < points.size(); i++)
    {
        if (vertices[i].z)
        {
            // upload the point and texture coordinates only for points we have depth data for
            glVertex3fv(vertices[i]);
            glTexCoord2fv(tex_coords[i]);
        }
    }

    // OpenGL cleanup
    glEnd();
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
}

// Registers the state variable and callbacks to allow mouse control of the pointcloud
void register_glfw_callbacks(window& app, glfw_state& app_state)
{
    app.on_left_mouse = [&](bool pressed)
    {
        app_state.ml = pressed;
    };

    app.on_mouse_scroll = [&](double xoffset, double yoffset)
    {
        app_state.offset_x -= static_cast<float>(xoffset);
        app_state.offset_y -= static_cast<float>(yoffset);
    };

    app.on_mouse_move = [&](double x, double y)
    {
        if (app_state.ml)
        {
            app_state.yaw -= (x - app_state.last_x);
            app_state.yaw = std::max(app_state.yaw, -120.0);
            app_state.yaw = std::min(app_state.yaw, +120.0);
            app_state.pitch += (y - app_state.last_y);
            app_state.pitch = std::max(app_state.pitch, -80.0);
            app_state.pitch = std::min(app_state.pitch, +80.0);
        }
        app_state.last_x = x;
        app_state.last_y = y;
    };

    app.on_key_release = [&](int key)
    {
        if (key == 32) // Escape
        {
            app_state.yaw = app_state.pitch = 0; app_state.offset_x = app_state.offset_y = 0.0;
        }
    };
}

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

void render_slider(rect location, float& clipping_dist)
{
    // Some trickery to display the control nicely
    static const int flags = ImGuiWindowFlags_NoCollapse
                             | ImGuiWindowFlags_NoScrollbar
                             | ImGuiWindowFlags_NoSavedSettings
                             | ImGuiWindowFlags_NoTitleBar
                             | ImGuiWindowFlags_NoResize
                             | ImGuiWindowFlags_NoMove;
    const int pixels_to_buttom_of_stream_text = 25;
    const float slider_window_width = 30;

    ImGui::SetNextWindowPos({ location.x, location.y + pixels_to_buttom_of_stream_text });
    ImGui::SetNextWindowSize({ slider_window_width + 20, location.h - (pixels_to_buttom_of_stream_text * 2) });

    //Render the vertical slider
    ImGui::Begin("slider", nullptr, flags);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    auto slider_size = ImVec2(slider_window_width / 2, location.h - (pixels_to_buttom_of_stream_text * 2) - 20);
    ImGui::VSliderFloat("", slider_size, &clipping_dist, 0.0f, 6.0f, "", 1.0f, true);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Depth Clipping Distance: %.3f", clipping_dist);
    ImGui::PopStyleColor(3);

    //Display bars next to slider
    float bars_dist = (slider_size.y / 6.0f);
    for (int i = 0; i <= 6; i++)
    {
        ImGui::SetCursorPos({ slider_size.x, i * bars_dist });
        std::string bar_text = "- " + std::to_string(6-i) + "m";
        ImGui::Text("%s", bar_text.c_str());
    }
    ImGui::End();
}

void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
    uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

    int width = other_frame.get_width();
    int height = other_frame.get_height();
    int other_bpp = other_frame.get_bytes_per_pixel();

#pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
    for (int y = 0; y < height; y++)
    {
        auto depth_pixel_index = y * width;
        for (int x = 0; x < width; x++, ++depth_pixel_index)
        {
            // Get the depth value of the current pixel
            auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

            // Check if the depth value is invalid (<=0) or greater than the threashold
            if (pixels_distance <= 0.f || pixels_distance > clipping_dist)
            {
                // Calculate the offset in other frame's buffer to current pixel
                auto offset = depth_pixel_index * other_bpp;

                // Set pixel to "background" color (0x999999)
                std::memset(&p_other_frame[offset], 0x99, other_bpp);
            }
        }
    }
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}
