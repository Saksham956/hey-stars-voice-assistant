"""Checked Windows default endpoint volume; ctypes only, no key injection.

Uses IMMDeviceEnumerator/IMMDevice/IAudioEndpointVolume. COM pointers are local
to the calling worker and released before return. No audio streams are opened.
"""
from contextlib import AbstractContextManager
import ctypes as ct
from dataclasses import dataclass
import os
import uuid


class GUID(ct.Structure):
    _fields_ = [('data1',ct.c_uint32),('data2',ct.c_uint16),
                ('data3',ct.c_uint16),('data4',ct.c_ubyte*8)]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


def checked(hr):
    if hr < 0:
        raise OSError('Windows endpoint operation failed (HRESULT 0x%08X)' % (hr & 0xffffffff))


def call(pointer, index, *types):
    table = ct.cast(pointer,ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(ct.c_long,ct.c_void_p,*types)(table[index])


class EndpointVolume(AbstractContextManager):
    def __init__(self):
        self.enumerator=ct.c_void_p()
        self.device=ct.c_void_p()
        self.endpoint=ct.c_void_p()
        self.initialized=False

    def __enter__(self):
        if os.name != 'nt': raise OSError('Volume control requires Windows')
        self.ole=ct.WinDLL('ole32')
        self.ole.CoInitializeEx.argtypes=[ct.c_void_p,ct.c_uint32]
        self.ole.CoInitializeEx.restype=ct.c_long
        self.ole.CoUninitialize.argtypes=[]
        self.ole.CoUninitialize.restype=None
        self.ole.CoCreateInstance.argtypes=[ct.POINTER(GUID),ct.c_void_p,ct.c_uint32,
                                            ct.POINTER(GUID),ct.POINTER(ct.c_void_p)]
        self.ole.CoCreateInstance.restype=ct.c_long
        try:
            hr=self.ole.CoInitializeEx(None,0)
            # A pre-existing STA can also use this interface; do not uninitialize
            # an apartment whose initialization count we did not increment.
            if (hr & 0xffffffff) != 0x80010106: checked(hr)
            self.initialized=hr >= 0
            clsid=GUID.parse('BCDE0395-E52F-467C-8E3D-C4579291692E')
            iid=GUID.parse('A95664D2-9614-4F35-A746-DE8DB63617E6')
            checked(self.ole.CoCreateInstance(ct.byref(clsid),None,1,ct.byref(iid),ct.byref(self.enumerator)))
            # Default rendering endpoint, eConsole: the Windows master control.
            checked(call(self.enumerator,4,ct.c_int,ct.c_int,ct.POINTER(ct.c_void_p))(
                self.enumerator,0,0,ct.byref(self.device)))
            iid=GUID.parse('5CDF2C82-841E-4546-9722-0CF74078229A')
            checked(call(self.device,3,ct.POINTER(GUID),ct.c_uint32,ct.c_void_p,ct.POINTER(ct.c_void_p))(
                self.device,ct.byref(iid),23,None,ct.byref(self.endpoint)))
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def level(self):
        level=ct.c_float()
        checked(call(self.endpoint,9,ct.POINTER(ct.c_float))(self.endpoint,ct.byref(level)))
        if not 0 <= level.value <= 1: raise OSError('Invalid Windows endpoint volume')
        return level.value

    def muted(self):
        muted=ct.c_int()
        checked(call(self.endpoint,15,ct.POINTER(ct.c_int))(self.endpoint,ct.byref(muted)))
        return bool(muted.value)

    def step(self, direction):
        # IUnknown (3 methods), notifications (2), then endpoint methods.
        index=17 if direction=='volume_up' else 18
        checked(call(self.endpoint,index,ct.c_void_p)(self.endpoint,None))

    def __exit__(self,*args):
        try:
            for pointer in (self.endpoint,self.device,self.enumerator):
                if pointer:
                    # IUnknown::Release returns ULONG, not HRESULT.
                    table=ct.cast(pointer,ct.POINTER(ct.POINTER(ct.c_void_p))).contents
                    ct.WINFUNCTYPE(ct.c_ulong,ct.c_void_p)(table[2])(pointer)
                    pointer.value=None
        finally:
            if self.initialized:
                self.ole.CoUninitialize()
                self.initialized=False


@dataclass(frozen=True)
class VolumeChange:
    before: float
    after: float
    muted: bool
    at_limit: bool = False


def change_volume(direction, *, is_current=lambda: True):
    if direction not in ('volume_up','volume_down'): raise ValueError('Invalid volume direction')
    if not is_current(): return None
    with EndpointVolume() as endpoint:
        before=endpoint.level()
        at_limit=before >= 1-1e-6 if direction=='volume_up' else before <= 1e-6
        if not is_current(): return None
        if not at_limit: endpoint.step(direction)
        after=endpoint.level()
        if not at_limit and not (after > before if direction=='volume_up' else after < before):
            raise OSError('Windows volume did not change in the requested direction')
        return VolumeChange(before,after,endpoint.muted(),at_limit)
