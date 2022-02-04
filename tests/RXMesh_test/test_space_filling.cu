#include "gtest/gtest.h"

namespace Hilbert 
{

// Based on Hacker's Delight 2nd edition
unsigned int distance_from(unsigned int x, unsigned int y, int curve_order)
{
    int      n = curve_order;
    int      i, xi, yi;
    unsigned s, temp;

    s = 0;  // Initialize.

    for (i = n - 1; i >= 0; i--) 
    {
        xi = (x >> i) & 1;  // Get bit i of x.
        yi = (y >> i) & 1;  // Get bit i of y.
        if (yi == 0) 
        {
            temp = x;             // Swap x and y and,
            x    = y ^ (-xi);     // if xi = 1,
            y    = temp ^ (-xi);  // complement them.
        }
        s = 4 * s + 2 * xi + (xi ^ yi);  // Append two bits to s.
    }

    return s;
}

// Based on Hacker's Delight 2nd edition
void point_from_distance(unsigned s, int order, unsigned* xp, unsigned* yp)
{
    int      n = order;
    int      i, sa, sb;
    unsigned x(0), y(0), temp(0);
    for (i = 0; i < 2 * n; i += 2) {
        sa = (s >> (i + 1)) & 1;  // Get bit i+1 of s.
        sb = (s >> i) & 1;        // Get bit i of s.
        if ((sa ^ sb) == 0) {     // If sa,sb = 00 or 11,
            temp = x;             // swap x and y,
            x    = y ^ (-sa);     // and if sa = 1,
            y    = temp ^ (-sa);  // complement them.
        }
        x = (x >> 1) | (sa << 31);         // Prepend sa to x and
        y = (y >> 1) | ((sa ^ sb) << 31);  // (sa^sb) to y.
    }
    *xp = x >> (32 - n);  // Right-adjust x and y
    *yp = y >> (32 - n);  // and return them to
}                         // the caller.

}

TEST(RXMesh, SpaceFilling)
{
    unsigned x = 1, y = 0;
    int      order = 1;
    std::cout << x << ", " << y << '\n';
    unsigned d = Hilbert::distance_from(x, y, order);
    std::cout << "d = " << d << '\n';
    EXPECT_TRUE(d == 3);

    unsigned xp=0, yp=0;
    Hilbert::point_from_distance(3, order, &xp, &yp);
    EXPECT_TRUE((xp == x) && (yp == y));


    std::cout << "======Hilbert ==========================\n";
    order = 3;
    unsigned maxd = UINT32_MAX;

    for (unsigned i = 0; i < 1 << order; i++) 
    {
        for (unsigned j = 0; j < 1 << order; j++) 
        {
           std::cout << "-------------\n";
           std::cout << "i = " << i << ", j = " << j << '\n';
            d = Hilbert::distance_from(i, j, order);
           std::cout << "d = " << d << '\n';

            Hilbert::point_from_distance(d, order, &xp, &yp);
            std::cout << "xp = " << xp << ", yp = " << yp << '\n';
            EXPECT_TRUE((xp == i) && (yp == j));
            maxd = std::max(maxd, d);
        }
    }
    std::cout << "maxd = " << maxd << std::endl;
    std::cout << "======End Hilbert ==========================\n";
}
