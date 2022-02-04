#include "gtest/gtest.h"
#include "rxmesh/space_filling.h"

TEST(RXMesh, SpaceFilling)
{
    using namespace rxmesh;

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
