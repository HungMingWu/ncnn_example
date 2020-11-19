#pragma once
#include <orbwebai/structure.h>

namespace orbwebai 
{
    template<typename _Tp> static inline
    Point_<_Tp> operator-(const Point_<_Tp>& a, const Point_<_Tp>& b)
    {
        return Point_<_Tp>(a.x - b.x, a.y - b.y);
    }

    template <typename _Tp>
    inline Point_<_Tp> Rect_<_Tp>::br() const
    {
        return Point_<_Tp>(x + width, y + height);
    }

    template <typename _Tp>
    inline _Tp Rect_<_Tp>::area() const
    {
        return width * height;
    }

    template<typename _Tp> static inline
    Rect_<_Tp>& operator &= (Rect_<_Tp>& a, const Rect_<_Tp>& b)
    {
        _Tp x1 = std::max(a.x, b.x);
        _Tp y1 = std::max(a.y, b.y);
        a.width = std::min(a.x + a.width, b.x + b.width) - x1;
        a.height = std::min(a.y + a.height, b.y + b.height) - y1;
        a.x = x1;
        a.y = y1;
        if (a.width <= 0 || a.height <= 0)
            a = Rect_< _Tp>();
        return a;
    }

    template<typename _Tp> static inline
    Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
    {
        Rect_<_Tp> c = a;
        return c &= b;
    }
}