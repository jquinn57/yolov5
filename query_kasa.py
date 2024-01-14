from kasa import SmartStrip
import asyncio

async def query_power(ip, plug_name):
    strip = SmartStrip(ip)
    await strip.update()
    plug = strip.get_plug_by_name(plug_name)
    while True:
        power = await plug.get_emeter_realtime()
        print(power)
        await asyncio.sleep(5)


if __name__ == '__main__':
    ip = '192.168.1.87'
    asyncio.run(query_power(ip, 'MX3'))
