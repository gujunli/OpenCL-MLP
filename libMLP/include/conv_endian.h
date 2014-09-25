#ifndef _CONV_ENDIAN_H_
#define _CONV_ENDIAN_H_

// conversion of word type data between host and assigned endian
static inline void LEtoLEHosts(unsigned short &x)
{
};

static inline void LEtoHosts(unsigned short &x)
{
    LEtoLEHosts(x);       // for x86 and x86-64
};

static inline void LEHostToLEs(unsigned short &x)
{
};

static inline void HostToLEs(unsigned short &x)
{
   	LEHostToLEs(x);  // for x86 architecture
	                 // ToDO: other architecture
};

static inline void BEtoLEHosts(unsigned short &x)
{
    x = ( (x >> 8) & 0x00ff ) | ( (x << 8) & 0xff00 );
};

static inline void BEtoHosts(unsigned short &x)
{
    BEtoLEHosts(x);      // for x86 and x86-64
};

static inline void LEHostToBEs(unsigned short &x)
{
    x = ( (x >> 8) & 0x00ff ) | ( (x << 8) & 0xff00 );
};

static inline void HostToBEs(unsigned short &x)
{
   	LEHostToBEs(x);  // for x86 architecture
	                 // ToDO: other architecture
};

// conversion of dword type data between host and assigned endian
static void BEtoLEHostl(unsigned int &x)
{
  x = ((x>>24) & 0x000000FF) | ((x<<8) & 0x00FF0000) | ((x>>8) & 0x0000FF00) | ((x<<24) & 0xFF000000);
};

static void BEtoHostl(unsigned int &x)
{
	BEtoLEHostl(x);  // for x86 architecture
};

static inline void LEHostToBEl(unsigned int &x)
{
  x = ((x>>24) & 0x000000FF) | ((x<<8) & 0x00FF0000) | ((x>>8) & 0x0000FF00) | ((x<<24) & 0xFF000000);
};

static inline void HostToBEl(unsigned int &x)
{
	LEHostToBEl(x);   // for x86 architecture
	                  // TODO: other architectures
};

static inline void LEHostToLEl(unsigned int &x)
{
};

static inline void HostToLEl(unsigned int &x)
{
   	LEHostToLEl(x);  // for x86 architecture
	                 // ToDO: other architecture
};

static inline void LEtoLEHostl(unsigned int &x)
{
};

static inline void LEtoHostl(unsigned int &x)
{
	LEtoLEHostl(x);  // for x86 architecture
                         // ToDO: other architecture
};


// Don't care what byte order it is on the host side
static inline void FloatToBytes(float &x)
{
    unsigned int *datap;
    unsigned char *cp;
    unsigned char byte0, byte1, byte2, byte3;

    datap = (unsigned int *) &x;
    byte0 = (*datap) & 0x000000ff;
    byte1 = ( (*datap) & 0x0000ff00 ) >> 8;
    byte2 = ( (*datap) & 0x00ff0000 ) >> 16;
    byte3 = ( (*datap) & 0xff000000 ) >> 24;

    cp = (unsigned char *) &x;
    *cp = byte0;
    *(cp+1) = byte1;
    *(cp+2) = byte2;
    *(cp+3) = byte3;
};

// Don't care what byte order it is on the host side
static inline void BytesToFloat(float &x)
{
    unsigned int *datap;
    unsigned char *cp;
    unsigned int byte0, byte1, byte2, byte3;

    cp = (unsigned char *) &x;
    byte0 = (*cp);
    byte1 = *(cp+1) << 8;
    byte2 = *(cp+2) << 16;
    byte3 = *(cp+3) << 24;

    datap = (unsigned int *) &x;
    *datap = byte0 | byte1 | byte2 | byte3;
};

#endif


